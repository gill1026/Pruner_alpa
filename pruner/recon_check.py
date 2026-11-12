"""Sanity check: can a decoder reconstruct the full image from ALL visual tokens (no Pruner, no masking)?

If this fails, the Pruner gets no useful gradient, so it must pass before joint training.
  python -m pruner.recon_check --decoder_type qwen --decoder_init llm_layers --name qwen_llm
"""
import argparse, json, os, time
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .data import WindowDataset, load_index, split_windows
from .encoder import FrozenEncoder
from .losses import recon_loss
from .model import build_decoder, decoder_param_groups

H = os.path.expanduser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pruner/config.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--decoder_type", default="qwen")
    ap.add_argument("--decoder_init", default="llm_layers")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--lr_pre", type=float, default=2e-5)
    ap.add_argument("--val_windows", type=int, default=16)
    ap.add_argument("--lam", type=float, default=None, help="SSIM weight override")
    ap.add_argument("--overfit", action="store_true", help="train and evaluate on one fixed batch (bug check)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"] |= dict(decoder_type=a.decoder_type, decoder_init=a.decoder_init)
    if a.lam is not None:
        cfg["loss"]["lam"] = a.lam
    P = {k: H(v) for k, v in cfg["paths"].items()}
    out = os.path.join(P["out_dir"], "recon_check", a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda"

    parts = torch.load(P["parts"], map_location="cpu", weights_only=False)
    train_rows, val_rows, _ = split_windows(load_index(P["data_dir"]), cfg["data"]["val_frac"], cfg["data"]["split_seed"])
    mk = lambda rows: WindowDataset(P["data_dir"], rows, parts["cams"], parts["frames"], parts["grid_hw"])
    bw = cfg["train"]["batch_windows"]
    train_dl = DataLoader(mk(train_rows), batch_size=bw, shuffle=True, drop_last=True, num_workers=6)
    val_dl = DataLoader(mk(val_rows[:a.val_windows]), batch_size=bw, num_workers=4)
    enc = FrozenEncoder(parts, P["processor_dir"], dev)
    dec = build_decoder(cfg["model"], parts).to(dev)
    pre, new = decoder_param_groups(dec)
    groups = [{"params": new, "lr": a.lr_new}] + ([{"params": pre, "lr": a.lr_pre}] if pre else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    print(f"{a.name}: decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.1f}M, pretrained tensors {len(pre)}, "
          f"lr_new {a.lr_new}, lr_pre {a.lr_pre}", flush=True)

    def step_fn(batch):
        if "v" in batch:  # cached encoder output (overfit mode)
            img, v = batch["img"], batch["v"]
        else:
            img = batch["images"].to(dev).flatten(0, 1)
            v = enc(img)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dec(v)
        gt = img.float() / 255.0
        loss, mse, ssim = recon_loss(pred, gt, cfg["loss"]["lam"])
        return loss.mean(), mse.mean().item(), ssim.mean().item(), pred, gt

    if a.overfit:
        b = next(iter(train_dl))
        img = b["images"].to(dev).flatten(0, 1)
        fixed = dict(img=img, v=enc(img))
        train_dl = val_dl = [fixed] * 25
    log, step, t0, acc = [], 0, time.time(), [0.0, 0.0]
    while step < a.steps:
        for batch in train_dl:
            loss, mse, ssim, _, _ = step_fn(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            acc[0] += mse; acc[1] += ssim
            if step % 25 == 0:
                rec = dict(step=step, mse=acc[0] / 25, ssim=acc[1] / 25, sec_per_step=(time.time() - t0) / 25)
                log.append(rec); print(json.dumps(rec), flush=True)
                acc, t0 = [0.0, 0.0], time.time()
            if step >= a.steps:
                break

    dec.eval()
    ms, ss, base, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(val_dl):
            _, mse, ssim, pred, gt = step_fn(batch)
            k = gt.shape[0]
            ms += mse * k; ss += ssim * k; n += k
            # reference: MSE of predicting each image's own mean color (what a blind decoder can reach)
            base += ((gt - gt.mean(dim=(2, 3), keepdim=True)) ** 2).mean().item() * k
            if bi == 0:
                idx = torch.linspace(0, k - 1, 4).long()
                g = torch.cat([torch.cat(list(gt[idx]), dim=2), torch.cat(list(pred[idx].float().clamp(0, 1)), dim=2)], dim=1)
                Image.fromarray((g.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{out}/viz.jpg", quality=90)
    res = dict(name=a.name, steps=a.steps, val_mse=ms / n, val_ssim=ss / n, mean_color_mse=base / n, train_log=log)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    print(f"RESULT {a.name}: val_mse {ms / n:.4f} val_ssim {ss / n:.3f} (mean-color baseline mse {base / n:.4f})", flush=True)


if __name__ == "__main__":
    main()
