"""Window-level dataset over the Grounded-SAM masking output (see masking/gsam_batch.py)."""
import json, os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

GROUPS = ("human", "vehicle", "traffic_sign", "barrier")  # road is foreground but not an "object"
BINS = ("le1tok", "1to4tok", "gt4tok")  # instance mask area vs one 32x32 token cell
TOKEN_PX = 32


def size_bin(area_px):
    return 0 if area_px <= TOKEN_PX ** 2 else 1 if area_px <= 4 * TOKEN_PX ** 2 else 2


def load_index(data_dir):
    return [json.loads(l) for l in open(os.path.join(data_dir, "index.jsonl"))]


def split_windows(index, val_frac=0.1, seed=0):
    """Clip-disjoint split. full256/test72 clips are evaluation-only and never used for training."""
    eval_clips = {r["clip_id"] for r in index if r["source"] in ("full256", "test72")}
    cand = [r for r in index if r["source"] == "dev72_shards" and r["clip_id"] not in eval_clips]
    clips = sorted({r["clip_id"] for r in cand})
    random.Random(seed).shuffle(clips)
    val_clips = set(clips[:max(1, round(len(clips) * val_frac))])
    train = [r for r in cand if r["clip_id"] not in val_clips]
    val = [r for r in cand if r["clip_id"] in val_clips]
    evalw = [r for r in index if r["source"] in ("full256", "test72")]
    return train, val, evalw


def load_exclude(data_dir, name="exclude_noroad.jsonl"):
    p = os.path.join(data_dir, name)
    if not os.path.exists(p):
        return set()
    return {(r["key"], r["file"]) for r in map(json.loads, open(p))}


class WindowDataset(Dataset):
    """One item = one Alpamayo input window: images ordered camera-major then frame, as the model sees them."""

    def __init__(self, data_dir, rows, cams, frames, grid_hw, exclude=frozenset()):
        self.dir, self.rows, self.cams, self.frames = data_dir, rows, list(cams), frames
        self.grid_hw, self.exclude = grid_hw, exclude

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        key = self.rows[i]["key"]
        d = os.path.join(self.dir, "windows", key)
        meta = {(im["cam"], im["frame"]): im for im in json.load(open(os.path.join(d, "meta.json")))["images"]}
        gh, gw = self.grid_hw
        imgs, fgs, valid = [], [], []
        obj = torch.zeros(len(self.cams) * self.frames, len(GROUPS), len(BINS), gh, gw, dtype=torch.bool)
        for cam in self.cams:
            for f in range(self.frames):
                im = meta[(cam, f)]
                stem = os.path.join(d, im["file"])
                imgs.append(torch.from_numpy(np.array(Image.open(stem + ".jpg").convert("RGB"))).permute(2, 0, 1))
                fgs.append(torch.from_numpy(np.array(Image.open(stem + "_fg.png")) > 127))
                valid.append((key, im["file"]) not in self.exclude)
                for inst in im["instances"]:
                    if inst["group"] in GROUPS and inst["token_cells"]:
                        cells = torch.tensor(inst["token_cells"])
                        obj[len(imgs) - 1, GROUPS.index(inst["group"]), size_bin(inst["area_px"]), cells[:, 0], cells[:, 1]] = True
        return dict(images=torch.stack(imgs), fg=torch.stack(fgs), valid=torch.tensor(valid), obj=obj, key=key)
