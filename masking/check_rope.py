"""Check MRoPE position ids that Alpamayo's Qwen3-VL assigns to the 24 input images."""
import sys, types, torch, json
sys.path.insert(0, "/home/aisl/alpamayo/alpamayo2/src")
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from alpamayo2_super.chat_template.conversation import construct_image

P = "/home/aisl/alpamayo/cosmos2b"
proc = AutoProcessor.from_pretrained(P, min_pixels=163840, max_pixels=196608)
cfg = AutoConfig.from_pretrained(P)
cams = torch.tensor([0, 1, 2, 3, 5, 6])
frames = torch.rand(6, 4, 3, 320, 576)

for cam_ids, frame_nums in [(True, False), (True, True), (False, False)]:
    try:
        content = construct_image({"image_frames": frames, "camera_indices": cams}, cam_ids, cams, frame_nums)
    except TypeError:
        content = construct_image(frames, cam_ids, cams, frame_nums)
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_vision_id=False)
    inp = proc(text=text, images=frames.flatten(0, 1), videos=None, return_tensors="pt", do_rescale=False)
    ids = inp["input_ids"]
    fake = types.SimpleNamespace(config=cfg)
    pos, _ = Qwen3VLModel.get_rope_index(fake, ids, inp["image_grid_thw"], None, inp["attention_mask"])
    is_img = ids[0] == cfg.image_token_id
    p = pos[:, 0, is_img].reshape(3, 24, -1)
    print(f"\n== include_camera_ids={cam_ids} include_frame_nums={frame_nums}")
    print("grid_thw", inp["image_grid_thw"][0].tolist(), "image tokens", int(is_img.sum()), "per image", p.shape[-1],
          "seq len", ids.shape[1])
    for k in [0, 1, 2, 3, 4, 5, 23]:
        t, h, w = p[:, k]
        print(f" img{k:2d} (cam {cams[k // 4].item()}, f{k % 4}): t {t.min().item()}..{t.max().item()}  "
              f"h {h.min().item()}..{h.max().item()}  w {w.min().item()}..{w.max().item()}")
    starts = p[0, :, 0]
    print(" start-id gaps between consecutive images:", (starts[1:] - starts[:-1]).tolist())
    # text between image 3 and image 4 (camera boundary)
    idx = torch.nonzero(is_img)[:, 0]
    a, b = idx[4 * p.shape[-1] - 1].item(), idx[4 * p.shape[-1]].item()
    print(" tokens between img3 and img4:", repr(proc.tokenizer.decode(ids[0, a + 1:b])))
    a, b = idx[p.shape[-1] - 1].item(), idx[p.shape[-1]].item()
    print(" tokens between img0 and img1:", repr(proc.tokenizer.decode(ids[0, a + 1:b])))
    print(" prefix before img0:", repr(proc.tokenizer.decode(ids[0, :idx[0]]))[-120:])
