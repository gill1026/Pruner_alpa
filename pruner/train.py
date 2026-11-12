"""Train ReconPruner: image -> frozen encoder -> Pruner -> reconstruction decoder, every step.

  python -m pruner.train --config pruner/config.yaml
  python -m pruner.train --config pruner/config.yaml --max_steps 20 --name smoke   # quick check
Re-running the same command resumes from <out_dir>/<name>/last.pt.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .data import WindowDataset, load_exclude, load_index, split_windows
from .encoder import FrozenEncoder
from .losses import total_loss
from .metrics import SelectionMetrics
from .model import ReconDecoder, ReconPruner, check_full_attention, ste_mask

H = os.path.expanduser


def forward_batch(batch, enc, pruner, decoder, cfg, dev):
    images = batch["images"].to(dev, non_blocking=True)  # (B, I, 3, H, W) uint8
    fg = batch["fg"].to(dev, non_blocking=True)  # (B, I, H, W) bool
    valid = batch["valid"].to(dev)
    b, i = images.shape[:2]
    v = enc(images.flatten(0, 1))  # (B*I, T, D), no grad
    t, d = v.shape[1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = pruner(v.reshape(b, i * t, d))  # (B, I*T)
        m_ste, _ = ste_mask(s.float())
        m_ste = m_ste.reshape(b * i, t, 1)
        pred_fore = decoder(m_ste * v)
        pred_back = decoder((1 - m_ste) * v)
    img = images.flatten(0, 1).float() / 255.0
    fgm = fg.flatten(0, 1)[:, None].float()
    gt_fore, gt_back = img * fgm, img * (1 - fgm)
    loss, logs = total_loss(pred_fore, pred_back, gt_fore, gt_back, valid.flatten(),
                            cfg["loss"]["alpha"], cfg["loss"]["lam"])
    logs["sel_ratio"] = (s > 0).float().mean().item()
    return loss, logs, dict(s=s, fg=fg, valid=valid, img=img, gt_fore=gt_fore, gt_back=gt_back,
                            pred_fore=pred_fore, pred_back=pred_back)


def save_viz(out, path, grid_hw, n=4):
    """Rows: original | selected tokens | GT fore | pred fore | GT back | pred back."""
    idx = torch.linspace(0, out["img"].shape[0] - 1, n).long()
    t = grid_hw[0] * grid_hw[1]
    sel = (out["s"].reshape(-1, t)[idx] > 0).reshape(n, 1, *grid_hw).float()
    sel = torch.nn.functional.interpolate(sel, scale_factor=32, mode="nearest")
    rows = [out["img"][idx], out["img"][idx] * (0.25 + 0.75 * sel), out["gt_fore"][idx],
            out["pred_fore"][idx].float().clamp(0, 1), out["gt_back"][idx], out["pred_back"][idx].float().clamp(0, 1)]
    g = torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)  # (3, 6H, nW)
    Image.fromarray((g.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(path, quality=90)


@torch.no_grad()
def validate(loader, enc, pruner, decoder, cfg, dev, viz_path, grid_hw):
    pruner.eval(); decoder.eval()
    met, sums, n = SelectionMetrics(), {}, 0
    for bi, batch in enumerate(loader):
        _, logs, out = forward_batch(batch, enc, pruner, decoder, cfg, dev)
        w = int(batch["valid"].sum())
        for k, x in logs.items():
            sums[k] = sums.get(k, 0.0) + x * w
        n += w
        met.update(out["s"], out["fg"], out["valid"], batch["obj"].to(dev))
        if bi == 0:
            save_viz(out, viz_path, grid_hw)
    pruner.train(); decoder.train()
    return {"val/" + k: x / max(n, 1) for k, x in sums.items()} | {"val/" + k: x for k, x in met.compute().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pruner/config.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--batch_windows", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.name:
        cfg["name"] = a.name
    if a.batch_windows:
        cfg["train"]["batch_windows"] = a.batch_windows
    P, T = {k: H(v) for k, v in cfg["paths"].items()}, cfg["train"]
    run = os.path.join(P["out_dir"], cfg["name"])
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(T["seed"])
    dev = "cuda"

    parts = torch.load(P["parts"], map_location="cpu", weights_only=False)
    grid_hw = parts["grid_hw"]
    index = load_index(P["data_dir"])
    train_rows, val_rows, _ = split_windows(index, cfg["data"]["val_frac"], cfg["data"]["split_seed"])
    excl = load_exclude(P["data_dir"], cfg["data"]["exclude"])
    mk = lambda rows: WindowDataset(P["data_dir"], rows, parts["cams"], parts["frames"], grid_hw, excl)
    nw = cfg["data"]["num_workers"]
    train_dl = DataLoader(mk(train_rows), batch_size=T["batch_windows"], shuffle=True, drop_last=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_dl = DataLoader(mk(val_rows), batch_size=T["batch_windows"], shuffle=False, num_workers=nw)

    enc = FrozenEncoder(parts, P["processor_dir"], dev)
    pruner = ReconPruner(parts, cfg["model"]["pruner_init"]).to(dev)
    d = parts["text_config"]["hidden_size"]
    decoder = ReconDecoder(d, grid_hw, cfg["model"]["decoder_width"], cfg["model"]["decoder_depth"],
                           cfg["model"]["decoder_heads"]).to(dev)
    check_full_attention(pruner, len(parts["img_idx"]), d, dev)

    opt = torch.optim.AdamW([{"params": pruner.parameters(), "lr": T["lr_pruner"]},
                             {"params": decoder.parameters(), "lr": T["lr_decoder"]}], weight_decay=T["weight_decay"])
    steps_per_epoch = len(train_dl)
    total = T["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / T["warmup_steps"]) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, st / max(1, total)))))
    step, best = 0, float("inf")
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        pruner.load_state_dict(ck["pruner"]); decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from {last} at step {step}", flush=True)

    npar = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    print(f"run {run}\ntrain windows {len(train_rows)}, val windows {len(val_rows)}, excluded images {len(excl)}\n"
          f"pruner {npar(pruner):.1f}M params, decoder {npar(decoder):.1f}M params\n"
          f"steps/epoch {steps_per_epoch}, total steps {total}", flush=True)
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=1)
    logf = open(os.path.join(run, "log.jsonl"), "a")

    def save(path, with_opt):
        ck = dict(pruner=pruner.state_dict(), decoder=decoder.state_dict(), step=step, best=best, config=cfg)
        if with_opt:
            ck |= dict(opt=opt.state_dict(), sched=sched.state_dict())
        torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)

    stop = min(total, a.max_steps) if a.max_steps else total
    t0, t_data, acc = time.time(), 0.0, {}
    pruner.train(); decoder.train()
    while step < stop:
        td = time.time()
        for batch in train_dl:
            t_data += time.time() - td
            loss, logs, _ = forward_batch(batch, enc, pruner, decoder, cfg, dev)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {logs}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pruner.parameters()) + list(decoder.parameters()), T["grad_clip"])
            opt.step(); sched.step(); step += 1
            for k, x in logs.items():
                acc[k] = acc.get(k, 0.0) + x
            if step % T["log_every"] == 0 or step == stop:
                n = T["log_every"] if step % T["log_every"] == 0 else step % T["log_every"]
                dt = time.time() - t0
                rec = {"step": step, "epoch": step / steps_per_epoch, "lr_pruner": sched.get_last_lr()[0],
                       "sec_per_step": dt / n, "data_frac": t_data / dt, **{k: x / n for k, x in acc.items()}}
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(" ".join(f"{k} {x:.4g}" for k, x in rec.items()), flush=True)
                t0, t_data, acc = time.time(), 0.0, {}
            if step % T["save_every_steps"] == 0:
                save(last, True)
            if step % (steps_per_epoch * T["val_every_epochs"]) == 0 or step == stop:
                ep = step / steps_per_epoch
                vm = validate(val_dl, enc, pruner, decoder, cfg, dev, os.path.join(run, f"viz_step{step:06d}.jpg"), grid_hw)
                vm |= {"step": step, "epoch": ep}
                logf.write(json.dumps(vm) + "\n"); logf.flush()
                print("VAL " + " ".join(f"{k} {x:.4g}" for k, x in vm.items() if "/" not in k[4:]), flush=True)
                if vm["val/loss"] < best:
                    best = vm["val/loss"]; save(os.path.join(run, "best.pt"), False)
                save(last, True)
                t0, t_data = time.time(), 0.0
            if step >= stop:
                break
            td = time.time()
    print(f"DONE step {step} best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
