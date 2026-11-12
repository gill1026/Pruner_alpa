"""Grid of sample images where Grounded-SAM found no road instance (t0 frame only)."""
import json, random, collections
import numpy as np
from PIL import Image
from gsam_window import overlay, grid

idx = [json.loads(l) for l in open("data/index.jsonl")]
cand = collections.defaultdict(list)
for r in idx:
    m = json.load(open("data/windows/" + r["key"] + "/meta.json"))
    for im in m["images"]:
        if im["frame"] == 3 and not any(i["group"] == "road" for i in im["instances"]):
            cand[im["cam_name"]].append((r["key"], im))
print({c: len(v) for c, v in cand.items()})
random.seed(1)
pick = []
for c, k in [("cross_right", 4), ("cross_left", 3), ("front_wide", 2), ("rear_left", 1), ("rear_right", 1), ("front_tele", 1)]:
    pick += random.sample(cand[c], k)
ovs, labs = [], []
for key, im in pick:
    d = "data/windows/" + key
    img = Image.open(f"{d}/{im['file']}.jpg").convert("RGB")
    masks = np.load(f"{d}/{im['file']}_inst.npz")["masks"]
    insts = [dict(i, _mask=mk) for i, mk in zip(im["instances"], masks)]
    ovs.append(overlay(img, insts))
    labs.append(f"{im['cam_name']}  {key[:8]}  fg {im['fg_ratio']:.2f}  inst {len(insts)}")
    print(labs[-1], key)
grid([ovs[i:i + 4] for i in range(0, 12, 4)], 576, 320, [labs[i:i + 4] for i in range(0, 12, 4)]).save(
    "out_noroad/grid_noroad.jpg", quality=90)
