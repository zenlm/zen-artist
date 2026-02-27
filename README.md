---
license: apache-2.0
language:
- en
tags:
- zen
- zen-lm
- image-generation
- image-editing
- diffusion
library_name: diffusers
pipeline_tag: text-to-image
---

<p align="center">
  <img src="https://zenlm.org/logo.png" width="300"/>
</p>

<h1 align="center">Zen Artist</h1>

<p align="center">
  <strong>Diffusion-based image generation and editing by Zen LM</strong>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/zenlm/zen-artist">HuggingFace</a> &nbsp;|&nbsp;
  📖 <a href="https://zenlm.org">Docs</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/zenlm">GitHub</a>
</p>

---

## Introduction

**Zen Artist** is a 7B-parameter diffusion-based image foundation model from Zen LM by Hanzo AI. It delivers strong performance in both **text-to-image generation** and **instruction-guided image editing**, with particular strength in precise text rendering and multi-image composition.

## Key Capabilities

- **Text-to-Image Generation**: Generate high-fidelity images from natural language prompts
- **Instruction-Based Editing**: Modify existing images using text instructions
- **Text Rendering**: Accurate in-image text, including multilingual characters
- **Multi-Image Composition**: Combine multiple reference images into a coherent scene
- **ControlNet Support**: Depth maps, edge maps, keypoint maps, and sketch conditions

## Model Specifications

| Attribute | Value |
|-----------|-------|
| Parameters | 7B |
| Architecture | MMDiT (diffusion-based) |
| Native Resolution | 1024 x 1024 |
| Maximum Resolution | 2048 x 2048 |
| Text Context | 32K tokens |
| Model ID | `zenlm/zen-artist` |
| License | Apache 2.0 |

## Quick Start

Install dependencies:

```bash
pip install transformers diffusers torch
```

### Text-to-Image Generation

```python
from diffusers import DiffusionPipeline
import torch

model_name = "zenlm/zen-artist"

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

prompt = "A cozy coffee shop at night, warm lighting, chalkboard sign with elegant lettering"
negative_prompt = " "

# Supported aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:2": (1584, 1056),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

image.save("output.png")
```

### Image Editing

```python
import torch
from PIL import Image
from diffusers import ZenArtistEditPipeline

pipeline = ZenArtistEditPipeline.from_pretrained(
    "zenlm/zen-artist",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

image = Image.open("./input.png").convert("RGB")
prompt = "Change the background to a sunset beach scene"

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("edited.png")
```

### Multi-Image Composition

```python
import torch
from PIL import Image
from diffusers import ZenArtistEditPlusPipeline

pipeline = ZenArtistEditPlusPipeline.from_pretrained(
    "zenlm/zen-artist",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

image1 = Image.open("person.jpg").convert("RGB")
image2 = Image.open("background.jpg").convert("RGB")

prompt = "Place the person naturally in the scenic background"

inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("composed.png")
```

## Multi-GPU Deployment

```bash
export NUM_GPUS_TO_USE=4
export TASK_QUEUE_SIZE=100
export TASK_TIMEOUT=300
```

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| Minimum (FP8) | 8GB | Reduced quality |
| Recommended (BF16) | 24GB | Full quality |
| Optimal (Multi-GPU) | 4x 24GB | High throughput |

## Community Support

Zen Artist is compatible with the HuggingFace `diffusers` ecosystem, including LoRA workflows, FP8 quantization, and layer-by-layer offloading for low-VRAM deployments.

## License

Apache 2.0

## Citation

```bibtex
@misc{zenlm2025zen-artist,
    title={Zen Artist: Diffusion-based Image Generation and Editing},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-artist}}
}
```

---

<p align="center">
  <strong>Zen LM by Hanzo AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/zenlm">HuggingFace</a> &nbsp;|&nbsp;
  <a href="https://github.com/zenlm">GitHub</a>
</p>
