"""Grounded-SAM masking for every Alpamayo input window under ~/son (FastDriveVLA nuScenes-FG style).

Per window dir  data/windows/<clip_id>_<t0_us>/ :
  cam{c}_f{f}.jpg        original image (as fed to the model, 576x320)
  cam{c}_f{f}_fg.png     binary foreground mask (0/255)
  cam{c}_f{f}_fore.jpg   foreground GT  = image * mask      (recon target for high-score tokens)
  cam{c}_f{f}_back.jpg   background GT  = image * (1-mask)  (recon target for low-score tokens)
  cam{c}_f{f}_inst.npz   per-instance masks (N,H,W) bool
  meta.json              timestamps, camera ids, per-image instances (class/score/box/area/token cells)
data/index.jsonl gets one line per finished window; finished windows are skipped on restart.
"""
import argparse, glob, io, json, os, shutil, time, traceback
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Sam2Model, Sam2Processor)
from gsam_window import CAM_NAMES, MODEL_CAMS, PROMPT, TOKEN_PX, map_label

H = os.path.expanduser
SOURCES = [  # order = processing order; first source wins for duplicate (clip_id, t0_us)
    ("full256", H("~/son/paper_eval/data/full256/frames/*.parquet")),
    ("test72", H("~/son/Alpamayo_MTP/data/test72.parquet")),
    ("dev72_shards", H("~/son/alpamayo_thor/data/dev72/frames/*.parquet")),
]
SETS = {  # window membership, recorded for later split management
    "full256": H("~/son/paper_eval/data/full256/full256_windows.json"),
    "dev72": H("~/son/alpamayo_thor/data/dev72/dev72_windows.json"),
}


def iter_windows():
    for src, pat in SOURCES:
        for f in sorted(glob.glob(pat)):
            for r in pq.read_table(f).to_pylist():
                yield src, os.path.relpath(f, H("~")), r


def mask_image(img, m, dev, a):
    dproc, dino, sproc, sam = m
    w, h = img.size
    with torch.inference_mode():
        inp = dproc(images=img, text=PROMPT, return_tensors="pt").to(dev)
        det = dproc.post_process_grounded_object_detection(
            dino(**inp), inp.input_ids, threshold=a.box_thr, text_threshold=a.text_thr,
            target_sizes=[(h, w)])[0]
        texts = det["text_labels"] if "text_labels" in det else det["labels"]
        keep = [(b, float(s), *map_label(l)) for b, s, l in
                zip(det["boxes"].cpu().numpy(), det["scores"].cpu().numpy(), texts)]
        keep = [k for k in keep if k[3] is not None]
        masks = np.zeros((0, h, w), bool)
        if keep:
            sin = sproc(images=img, input_boxes=[[k[0].tolist() for k in keep]], return_tensors="pt").to(dev)
            sout = sam(**sin, multimask_output=False)
            masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0][:, 0].numpy() > 0
    insts, fg = [], np.zeros((h, w), bool)
    for k, mk in zip(keep, masks):
        b = [float(x) for x in k[0]]
        ys, xs = np.nonzero(mk)
        cells = sorted({(int(y) // TOKEN_PX, int(x) // TOKEN_PX) for y, x in zip(ys, xs)})
        insts.append(dict(label=k[2], group=k[3], score=k[1], box=b, box_wh=[b[2] - b[0], b[3] - b[1]],
                          area_px=int(mk.sum()), area_ratio=float(mk.mean()),
                          n_token_cells=len(cells), token_cells=cells))
        fg |= mk
    return fg, insts, masks


def do_window(src, shard, r, m, dev, a):
    cams = r["camera_indices"]
    nf = r.get("num_frames_per_camera") or len(r["jpegs"]) // len(cams)
    key = f'{r["clip_id"]}_{r["t0_us"]}'
    tmp, dst = f"{a.out}/windows/.tmp_{key}", f"{a.out}/windows/{key}"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    images = []
    for ci, cam in enumerate(cams):
        if cam not in MODEL_CAMS:
            continue
        for f in range(nf):
            k = ci * nf + f
            raw = r["jpegs"][k]
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            fg, insts, masks = mask_image(img, m, dev, a)
            stem = f"cam{cam}_f{f}"
            with open(f"{tmp}/{stem}.jpg", "wb") as fh:
                fh.write(raw)  # keep original bytes, no re-encode
            arr = np.asarray(img)
            Image.fromarray((fg * 255).astype(np.uint8)).save(f"{tmp}/{stem}_fg.png")
            Image.fromarray(arr * fg[..., None]).save(f"{tmp}/{stem}_fore.jpg", quality=95)
            Image.fromarray(arr * ~fg[..., None]).save(f"{tmp}/{stem}_back.jpg", quality=95)
            np.savez_compressed(f"{tmp}/{stem}_inst.npz", masks=masks)
            rel = r.get("relative_timestamps")
            images.append(dict(file=stem, cam=cam, cam_name=CAM_NAMES[cam], frame=f,
                               rel_ts=rel[k] if rel else None, abs_ts=r["absolute_timestamps"][k],
                               size=list(img.size), fg_ratio=float(fg.mean()), instances=insts))
    meta = dict(clip_id=r["clip_id"], t0_us=r["t0_us"], source=src, shard=shard, maneuver=r.get("maneuver"),
                camera_indices_model=list(MODEL_CAMS), num_frames=nf, prompt=PROMPT, dino=a.dino, sam=a.sam,
                box_thr=a.box_thr, text_thr=a.text_thr, images=images)
    json.dump(meta, open(f"{tmp}/meta.json", "w"))
    os.sync()  # files hit disk before the window becomes visible (Thor rebooted mid-run once)
    shutil.rmtree(dst, ignore_errors=True); os.rename(tmp, dst)
    return key, len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=H("~/gil/data"))
    ap.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--box_thr", type=float, default=0.30)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new windows (testing)")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/windows", exist_ok=True)
    idx_path = f"{a.out}/index.jsonl"
    done = set()
    if os.path.exists(idx_path):
        for l in open(idx_path, "rb"):
            l = l.strip(b"\x00 \r\n")
            if l:
                done.add(json.loads(l)["key"])
    member, maneuver = {}, {}
    for name, p in SETS.items():
        for w in json.load(open(p))["windows"]:
            k = f'{w["clip_id"]}_{w["t0_us"]}'
            member.setdefault(k, []).append(name)
            if w.get("maneuver"):
                maneuver[k] = w["maneuver"]

    dev = "cuda"
    m = (AutoProcessor.from_pretrained(a.dino),
         AutoModelForZeroShotObjectDetection.from_pretrained(a.dino).to(dev).eval(),
         Sam2Processor.from_pretrained(a.sam),
         Sam2Model.from_pretrained(a.sam).to(dev).eval())

    seen, n_new, t_start = set(), 0, time.time()
    for src, shard, r in iter_windows():
        key = f'{r["clip_id"]}_{r["t0_us"]}'
        r.setdefault("maneuver", maneuver.get(key))
        if r["maneuver"] is None:
            r["maneuver"] = maneuver.get(key)
        if key in seen:
            continue
        seen.add(key)
        if key in done:
            continue
        t = time.time()
        try:
            key, n_img = do_window(src, shard, r, m, dev, a)
        except Exception:
            print(f"[ERR] {key}\n{traceback.format_exc()}", flush=True)
            continue
        dt = time.time() - t
        with open(idx_path, "a") as fh:
            fh.write(json.dumps(dict(key=key, clip_id=r["clip_id"], t0_us=r["t0_us"], source=src, shard=shard,
                                     sets=member.get(key, []), maneuver=r.get("maneuver"),
                                     n_images=n_img, sec=round(dt, 2))) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        n_new += 1
        print(f"[{len(done) + n_new}] {key} {src} {n_img} img {dt:.1f}s  "
              f"elapsed {(time.time() - t_start) / 3600:.2f}h", flush=True)
        if a.limit and n_new >= a.limit:
            break
    print(f"DONE new={n_new} total={len(done) + n_new} unique_seen={len(seen)} "
          f"hours={(time.time() - t_start) / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
