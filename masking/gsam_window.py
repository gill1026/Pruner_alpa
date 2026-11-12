"""Grounded-SAM (Grounding DINO + SAM 2.1) masking for one Alpamayo input window.

One window = (clip_id, t0_us): 6 model cameras x 4 frames, read from a frames parquet.
Saves per image: original jpg, binary fg mask, per-instance masks/boxes/classes/sizes (json+npz),
plus overlay grids for visual inspection.
"""
import argparse, io, json, os, time
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Sam2Model, Sam2Processor)

# FastDriveVLA foreground: humans, roads, vehicles, traffic signs (incl. lights), traffic barriers.
CLASSES = {
    "person": "human", "pedestrian": "human", "cyclist": "human",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
    "traffic light": "traffic_sign", "traffic sign": "traffic_sign",
    "traffic cone": "barrier", "barrier": "barrier",
    "road": "road",
}
PROMPT = " . ".join(CLASSES) + " ."
MODEL_CAMS = (0, 1, 2, 3, 5, 6)  # DRIVING_SIX_CAMERA_FOUR_FRAME
CAM_NAMES = {0: "cross_left", 1: "front_wide", 2: "cross_right", 3: "rear_left",
             4: "rear_tele", 5: "rear_right", 6: "front_tele"}
COLORS = {"human": (255, 40, 40), "vehicle": (40, 120, 255), "traffic_sign": (255, 210, 0),
          "barrier": (255, 120, 0), "road": (60, 200, 90)}
TOKEN_PX = 32  # patch 16 x spatial_merge 2


def map_label(text):
    text = text.strip().lower()
    if text in CLASSES:
        return text, CLASSES[text]
    for k in sorted(CLASSES, key=len, reverse=True):  # DINO may return merged phrases
        if k in text:
            return k, CLASSES[k]
    return text, None


def load_window(frames_dir, shard, clip_id, t0_us):
    t = pq.read_table(os.path.join(frames_dir, shard)).to_pylist()
    r = next(x for x in t if x["clip_id"] == clip_id and x["t0_us"] == t0_us)
    n = r["num_frames_per_camera"]
    out = []
    for ci, cam in enumerate(r["camera_indices"]):
        if cam not in MODEL_CAMS:
            continue
        for f in range(n):
            k = ci * n + f
            out.append(dict(cam=cam, frame=f, rel_ts=r["relative_timestamps"][k],
                            abs_ts=r["absolute_timestamps"][k],
                            img=Image.open(io.BytesIO(r["jpegs"][k])).convert("RGB")))
    return out


def overlay(img, insts):
    base = np.asarray(img).astype(np.float32)
    for inst in sorted(insts, key=lambda d: -d["area_px"]):  # small on top
        c = np.array(COLORS[inst["group"]], np.float32)
        m = inst["_mask"]
        base[m] = base[m] * 0.45 + c * 0.55
    im = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for inst in insts:
        if inst["group"] == "road":
            continue
        x0, y0, x1, y1 = inst["box"]
        d.rectangle([x0, y0, x1, y1], outline=COLORS[inst["group"]], width=1)
        d.text((x0 + 1, max(0, y0 - 10)), f'{inst["label"]} {inst["score"]:.2f}', fill=(255, 255, 255))
    return im


def grid(rows, cell_w, cell_h, labels):
    W, H = cell_w * len(rows[0]), (cell_h + 14) * len(rows)
    g = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(g)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            g.paste(im.resize((cell_w, cell_h)), (j * cell_w, i * (cell_h + 14) + 14))
            d.text((j * cell_w + 3, i * (cell_h + 14) + 1), labels[i][j], fill=(230, 230, 230))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--t0_us", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--box_thr", type=float, default=0.30)
    ap.add_argument("--text_thr", type=float, default=0.25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    T = {}

    t = time.time()
    dproc = AutoProcessor.from_pretrained(a.dino)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(a.dino).to(dev).eval()
    sproc = Sam2Processor.from_pretrained(a.sam)
    sam = Sam2Model.from_pretrained(a.sam).to(dev).eval()
    torch.cuda.synchronize(); T["model_load_s"] = time.time() - t

    t = time.time()
    items = load_window(a.frames_dir, a.shard, a.clip_id, a.t0_us)
    T["data_load_s"] = time.time() - t

    per_img = []
    t_all = time.time()
    for it in items:
        img = it["img"]; w, h = img.size
        t1 = time.time()
        with torch.inference_mode():
            inp = dproc(images=img, text=PROMPT, return_tensors="pt").to(dev)
            out = dino(**inp)
            det = dproc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=a.box_thr, text_threshold=a.text_thr,
                target_sizes=[(h, w)])[0]
        torch.cuda.synchronize(); t_dino = time.time() - t1
        boxes = det["boxes"].cpu().numpy(); scores = det["scores"].cpu().numpy()
        texts = det.get("text_labels", det.get("labels"))
        keep = [(b, s, *map_label(l)) for b, s, l in zip(boxes, scores, texts)]
        keep = [k for k in keep if k[3] is not None]

        t2 = time.time()
        masks = np.zeros((0, h, w), bool)
        if keep:
            with torch.inference_mode():
                sin = sproc(images=img, input_boxes=[[k[0].tolist() for k in keep]], return_tensors="pt").to(dev)
                sout = sam(**sin, multimask_output=False)
                masks = sproc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu())[0][:, 0].numpy() > 0
        torch.cuda.synchronize(); t_sam = time.time() - t2

        insts, fg = [], np.zeros((h, w), bool)
        for k, m in zip(keep, masks):
            b = [float(x) for x in k[0]]
            area = int(m.sum())
            ys, xs = np.nonzero(m)
            # Qwen3-VL token grid: one merged token per 32x32 px (576x320 -> 18x10)
            cells = sorted({(int(y) // TOKEN_PX, int(x) // TOKEN_PX) for y, x in zip(ys, xs)}) if area else []
            insts.append(dict(label=k[2], group=k[3], score=float(k[1]), box=b,
                              box_wh=[b[2] - b[0], b[3] - b[1]], area_px=area, area_ratio=area / (h * w),
                              n_token_cells=len(cells), token_cells=cells, _mask=m))
            fg |= m

        stem = f'cam{it["cam"]}_{CAM_NAMES[it["cam"]]}_f{it["frame"]}'
        img.save(f"{a.out}/{stem}.jpg", quality=95)
        Image.fromarray((fg * 255).astype(np.uint8)).save(f"{a.out}/{stem}_fg.png")
        if insts:
            np.savez_compressed(f"{a.out}/{stem}_inst.npz", masks=np.stack([i["_mask"] for i in insts]))
        meta = dict(cam=it["cam"], cam_name=CAM_NAMES[it["cam"]], frame=it["frame"], rel_ts=it["rel_ts"],
                    abs_ts=it["abs_ts"], size=[w, h], fg_ratio=float(fg.mean()), t_dino_s=t_dino, t_sam_s=t_sam,
                    instances=[{k: v for k, v in i.items() if k != "_mask"} for i in insts])
        json.dump(meta, open(f"{a.out}/{stem}.json", "w"), indent=1)
        it.update(fg=fg, insts=insts, ov=overlay(img, insts), meta=meta)
        per_img.append(meta)
        print(f'{stem}: {len(insts)} inst, fg {fg.mean():.2f}, dino {t_dino:.2f}s sam {t_sam:.2f}s', flush=True)
    torch.cuda.synchronize(); T["masking_total_s"] = time.time() - t_all
    T["per_image_mean_s"] = T["masking_total_s"] / len(items)
    T["dino_mean_s"] = float(np.mean([m["t_dino_s"] for m in per_img]))
    T["sam_mean_s"] = float(np.mean([m["t_sam_s"] for m in per_img]))

    # grids: rows = cameras, cols = frames (t-3 .. t0)
    cams = sorted({it["cam"] for it in items})
    by = {(it["cam"], it["frame"]): it for it in items}
    nf = max(it["frame"] for it in items) + 1
    labels = [[f'{CAM_NAMES[c]}  t{f - nf + 1 if f < nf - 1 else "0"} ({by[(c, f)]["rel_ts"]:+.2f}s)' for f in range(nf)] for c in cams]
    w, h = items[0]["img"].size
    grid([[by[(c, f)]["ov"] for f in range(nf)] for c in cams], w, h, labels).save(f"{a.out}/grid_overlay.jpg", quality=90)
    grid([[Image.fromarray((by[(c, f)]["fg"] * 255).astype(np.uint8)).convert("RGB") for f in range(nf)] for c in cams],
         w // 2, h // 2, labels).save(f"{a.out}/grid_fgmask.jpg", quality=90)
    grid([[by[(c, f)]["img"] for f in range(nf)] for c in cams], w // 2, h // 2, labels).save(f"{a.out}/grid_original.jpg", quality=90)

    summary = dict(clip_id=a.clip_id, t0_us=a.t0_us, prompt=PROMPT, dino=a.dino, sam=a.sam,
                   box_thr=a.box_thr, text_thr=a.text_thr, n_images=len(items),
                   gpu=torch.cuda.get_device_name(0), timing=T,
                   counts={g: sum(1 for m in per_img for i in m["instances"] if i["group"] == g) for g in COLORS})
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
