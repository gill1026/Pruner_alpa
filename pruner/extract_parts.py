"""Extract the frozen pieces of Alpamayo that Pruner training needs into one small file.

Run once on the machine that has the Alpamayo checkpoint and source. Saves:
  visual_state   vlm.model.visual.* (frozen vision encoder + spatial merger)
  layer0_state   vlm.model.language_model.layers.0.* (PrunerLayer init)
  prefix_ids     token ids of the real prompt from its start to the last <|vision_end|>
  pos_ids        MRoPE position ids (3, L) the LLM assigns to that prefix
  img_idx        positions of the image tokens inside the prefix (cam-major, frame, row, col)
  text_embeds    LLM input embeddings of the prefix (zeros at image positions)
so that training depends only on this file and `transformers`, not on the Alpamayo source tree.
"""
import argparse, json, os, sys, types
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

H = os.path.expanduser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=H("~/alpamayo/ckpt/s2_final_G/final.pt"))
    ap.add_argument("--alpamayo_src", default=H("~/alpamayo/alpamayo2/src"))
    ap.add_argument("--backbone_dir", default=H("~/alpamayo/cosmos2b"), help="Qwen3-VL config + processor")
    ap.add_argument("--teacher_cfg", default=H("~/alpamayo/data/coc_34b_v1/_teacher_assets/config.json"))
    ap.add_argument("--cams", default="0,1,2,3,5,6")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--out", default=H("~/gil/parts/alpamayo_parts.pt"))
    a = ap.parse_args()

    sys.path.insert(0, a.alpamayo_src)
    from alpamayo2_super.chat_template.conversation import construct_image, construct_system_prompt

    t_cfg = json.load(open(a.teacher_cfg))
    cam_ids, frame_nums = bool(t_cfg["include_camera_ids"]), t_cfg["frame_label"] == "frame_num"
    vlm_cfg = json.load(open(os.path.join(a.backbone_dir, "config.json")))
    proc = AutoProcessor.from_pretrained(a.backbone_dir, min_pixels=t_cfg["min_pixels"], max_pixels=t_cfg["max_pixels"])

    cams = torch.tensor([int(c) for c in a.cams.split(",")])
    frames = torch.rand(len(cams), a.frames, 3, a.height, a.width)
    msgs = [{"role": "system", "content": construct_system_prompt()},
            {"role": "user", "content": construct_image({"image_frames": frames}, cam_ids, cams, frame_nums)}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, add_vision_id=False)
    inp = proc(text=text, images=frames.flatten(0, 1), videos=None, return_tensors="pt", do_rescale=False)
    ids = inp["input_ids"][0]

    # image is the first user component in Alpamayo's prompt, so everything up to the last
    # <|vision_end|> is identical to the real prompt regardless of what follows
    end = int(torch.nonzero(ids == vlm_cfg["vision_end_token_id"])[-1]) + 1
    fake = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=vlm_cfg["vision_config"]["spatial_merge_size"]),
        image_token_id=vlm_cfg["image_token_id"], video_token_id=vlm_cfg["video_token_id"],
        vision_start_token_id=vlm_cfg["vision_start_token_id"]))
    pos, _ = Qwen3VLModel.get_rope_index(fake, inp["input_ids"], inp["image_grid_thw"], None, inp["attention_mask"])
    ids, pos = ids[:end], pos[:, 0, :end]
    is_img = ids == vlm_cfg["image_token_id"]
    n_img = len(cams) * a.frames
    tok_per_img = int(is_img.sum()) // n_img
    assert int(is_img.sum()) == n_img * tok_per_img

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    vis = {k[len("vlm.model.visual."):]: v for k, v in sd.items() if k.startswith("vlm.model.visual.")}
    l0p = "vlm.model.language_model.layers.0."
    layer0 = {k[len(l0p):]: v for k, v in sd.items() if k.startswith(l0p)}
    emb = sd["vlm.model.language_model.embed_tokens.weight"][ids].clone()
    emb[is_img] = 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(
        visual_state=vis, layer0_state=layer0, vision_config=vlm_cfg["vision_config"], text_config=vlm_cfg["text_config"],
        prefix_ids=ids, pos_ids=pos, img_idx=torch.nonzero(is_img)[:, 0], text_embeds=emb,
        cams=cams.tolist(), frames=a.frames, height=a.height, width=a.width, tok_per_img=tok_per_img,
        grid_hw=[a.height // 32, a.width // 32], min_pixels=t_cfg["min_pixels"], max_pixels=t_cfg["max_pixels"],
        include_camera_ids=cam_ids, include_frame_nums=frame_nums, source_ckpt=a.ckpt), a.out)
    print(f"saved {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB)")
    print(f"visual tensors {len(vis)}, layer0 tensors {len(layer0)}, prefix len {end}, "
          f"image tokens {int(is_img.sum())} ({n_img} x {tok_per_img}), text tokens {end - int(is_img.sum())}")
    print("prefix head:", repr(proc.tokenizer.decode(ids[:int(torch.nonzero(is_img)[0])]))[-200:])
    print("pos range:", pos.min().item(), "..", pos.max().item())


if __name__ == "__main__":
    main()
