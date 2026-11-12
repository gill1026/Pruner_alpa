"""Frozen Alpamayo vision encoder (Qwen3-VL ViT + spatial merger), loaded from extracted parts."""
import torch
from transformers import AutoImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class FrozenEncoder:
    def __init__(self, parts, processor_dir, device):
        self.device = device
        self.tok_per_img = parts["tok_per_img"]
        self.visual = Qwen3VLVisionModel(Qwen3VLVisionConfig(**parts["vision_config"]))
        # strict: every vision weight must come from the Alpamayo checkpoint
        self.visual.load_state_dict(parts["visual_state"], strict=True)
        self.visual.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.proc = AutoImageProcessor.from_pretrained(
            processor_dir, min_pixels=parts["min_pixels"], max_pixels=parts["max_pixels"], use_fast=True)

    @torch.no_grad()
    def __call__(self, images):
        """images: uint8 (N, 3, H, W) on device -> merger output (N, tok_per_img, D), float32."""
        # same call as Alpamayo's helper.prepare_model_inputs: float [0,1] images, do_rescale=False
        out = self.proc(images=images.float() / 255.0, do_rescale=False, return_tensors="pt", device=self.device)
        pix, grid = out["pixel_values"].to(self.device), out["image_grid_thw"].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb, _deepstack = self.visual(pix.to(torch.bfloat16), grid_thw=grid)
        emb = emb.float().reshape(images.shape[0], -1, emb.shape[-1])
        assert emb.shape[1] == self.tok_per_img, (emb.shape, self.tok_per_img)
        return emb
