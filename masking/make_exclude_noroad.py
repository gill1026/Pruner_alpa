"""List images with no road instance -> data/exclude_noroad.jsonl (excluded from Pruner training loss).

Files are not moved: a window must keep all 24 images to stay a valid Alpamayo input.
"""
import json, collections

idx = [json.loads(l) for l in open("data/index.jsonl")]
evalc = {r["clip_id"] for r in idx if r["source"] in ("full256", "test72")}
n_img = collections.Counter(); n_exc = collections.Counter(); per_cam = collections.Counter()
win_exc = collections.Counter(); rows = []
for r in idx:
    split = "eval" if r["source"] in ("full256", "test72") else ("eval_clip_overlap" if r["clip_id"] in evalc else "train_candidate")
    m = json.load(open("data/windows/" + r["key"] + "/meta.json"))
    for im in m["images"]:
        n_img[split] += 1
        if not any(i["group"] == "road" for i in im["instances"]):
            rows.append(dict(key=r["key"], file=im["file"], cam=im["cam"], cam_name=im["cam_name"], frame=im["frame"],
                             split=split, fg_ratio=im["fg_ratio"], n_instances=len(im["instances"])))
            n_exc[split] += 1; per_cam[im["cam_name"]] += 1; win_exc[(split, r["key"])] += 1
with open("data/exclude_noroad.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
print("excluded total", len(rows), "of", sum(n_img.values()))
for s in n_img:
    w = [v for (sp, _), v in win_exc.items() if sp == s]
    print(s, "images", n_img[s], "excluded", n_exc[s], "(%.1f%%)" % (100 * n_exc[s] / n_img[s]),
          "| windows touched", len(w), "fully excluded (24/24)", sum(v == 24 for v in w))
print("per cam", dict(per_cam))
