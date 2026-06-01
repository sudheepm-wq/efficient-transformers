# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
WAN 720P Lightning inference on 24 Qualcomm AI 100 devices using device sharing.

Device layout (defined in wan_non_unified_720p_config.json):
  transformer_high  ->  devices [0-15]   (high-noise denoising steps, then unloaded)
  transformer_low   ->  devices [0-15]   (low-noise  denoising steps, loaded after high is done)
  vae_decoder       ->  devices [16-23]

The share_transformer_devices=True toggle enables this sequential device reuse,
allowing the 720P model to run on 24 devices instead of 40.

Usage:
  python wan_lightning_720p.py
"""

import os

import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# Path to the 720P device-sharing config (same directory as this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_SCRIPT_DIR, "wan_non_unified_720p_config.json")

# Load the pipeline in non-unified mode with device sharing enabled.
# share_transformer_devices=True: transformer_high is loaded first, runs all
# high-noise steps, then is unloaded; transformer_low is loaded on the same
# devices and runs all low-noise steps.
pipeline = QEffWanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    use_unified=False,
    share_transformer_devices=True,
)

# Download the Lightning LoRA weights
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)


def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# Attach LoRA adapters
pipeline.transformer.model.transformer_high.load_lora_adapter(
    load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
)
pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])
pipeline.transformer.model.transformer_low.load_lora_adapter(
    load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
)
pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])

prompt = (
    "In a warmly lit living room, an elderly man with gray hair sits in a wooden armchair "
    "adorned with a blue cushion. He wears a gray cardigan over a white shirt, engrossed in "
    "reading a book. As he turns the pages, he subtly adjusts his posture, ensuring his glasses "
    "stay in place. He then removes his glasses, holding them in his hand, and turns his head to "
    "the right, maintaining his grip on the book. The soft glow of a bedside lamp bathes the "
    "scene, creating a calm and serene atmosphere, with gentle shadows enhancing the intimate setting."
)

output = pipeline(
    prompt=prompt,
    height=720,
    width=1280,
    num_frames=81,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(0),
    use_onnx_subfunctions=True,
    parallel_compile=True,
    # wan_non_unified_720p_config.json assigns:
    #   transformer_high -> device_ids [0-15]
    #   transformer_low  -> device_ids [0-15]  (same devices, loaded after high is unloaded)
    #   vae_decoder      -> device_ids [16-23]
    custom_config_path=CONFIG_PATH,
)

frames = output.images[0]
export_to_video(frames, "output_t2v_720p.mp4", fps=16)
print(output)