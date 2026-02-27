<p align="center">
    <img src="https://zenlm.org/logo.png" width="400"/>
<p> 
<p align="center">&nbsp&nbsp💜 <a href="https://hanzo.chat/">Zen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/zenlm/Zen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/zenlm/Zen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://zenlm.org/blog/zen-artist">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://zenlm.org/blog/zen-artist">Blog(Edit)</a> &nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/zenlm/zen">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/zenlm/zen">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://zenlm.org/logo.png" width="1024"/>
<p>

## Introduction
We are thrilled to release **Zen-Image**, a 20B MMDiT image foundation model that achieves significant advances in **complex text rendering** and **precise image editing**. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.


![](https://zenlm.org/logo.png)

## News
- 2025.09.22: This September, we are pleased to introduce Zen-Image-Edit-2509, the monthly iteration of Zen-Image-Edit. To experience the latest model, please visit [Zen Chat](https://hanzo.chat)  and select the "Image Editing" feature. Compared with Zen-Image-Edit released in August, the main improvements of Zen-Image-Edit-2509 include:
![](https://zenlm.org/logo.png)
  - **Multi-image Editing Support**: For multi-image inputs, Zen-Image-Edit-2509 builds upon the Zen-Image-Edit architecture and is further trained via image concatenation to enable multi-image editing. It supports various combinations such as "person + person," "person + product," and "person + scene." Optimal performance is currently achieved with 1 to 3 input images.

  - **Enhanced Single-image Consistency**: For single-image inputs, Zen-Image-Edit-2509 significantly improves consistency, specifically in the following areas:
    - **Improved Person Editing Consistency**: Better preservation of facial identity, supporting various portrait styles and pose transformations;
    - **Improved Product Editing Consistency**: Better preservation of product identity, supporting product poster editing；
    - **Improved Text Editing Consistency**: In addition to modifying text content, it also supports editing text fonts, colors, and materials；

  - **Native Support for ControlNet**: Including depth maps, edge maps, keypoint maps, and more.


- 2025.08.19: We have observed performance misalignments of Zen-Image-Edit. To ensure optimal results, please update to the latest diffusers commit. Improvements are expected, especially in identity preservation and instruction following.
- 2025.08.18: We’re excited to announce the open-sourcing of Zen-Image-Edit! 🎉 Try it out in your local environment with the quick start guide below, or head over to [Zen Chat](https://hanzo.chat/) or [Huggingface Demo](https://huggingface.co/spaces/zenlm/zen) to experience the online demo right away! If you enjoy our work, please show your support by giving our repository a star. Your encouragement means a lot to us!
- 2025.08.09: Zen-Image now supports a variety of LoRA models, such as MajicBeauty LoRA, enabling the generation of highly realistic beauty images.
![](https://zenlm.org/logo.png)
    
- 2025.08.05: Zen-Image is now natively supported in ComfyUI, see [Zen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/zen-artist)
- 2025.08.05: Zen-Image is now on Zen Chat. Click [Zen Chat](https://hanzo.chat/) and choose "Image Generation".
- 2025.08.05: We released our [Technical Report](https://arxiv.org/abs/2508.02324) on Arxiv!
- 2025.08.04: We released Zen-Image weights! Check at [Huggingface](https://huggingface.co/zenlm/Zen-Image)!
- 2025.08.04: We released Zen-Image! Check our [Blog](https://zenlm.org/blog/zen-artist) for more details!

## Quick Start

1. Make sure your transformers>=4.51.3 (Supporting zen-VL)

2. Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

### Zen-Image (for Text-to-Image)

The following contains a code snippet illustrating how to use the model to generate images based on text prompts:

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Zen/Zen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Zen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

### Zen-Image-Edit (for Image Editing, Only Support Single Image Input)
> [!NOTE]
> Zen-Image-Edit-2509 has better consistency than Zen-Image-Edit; it is recommended to use Zen-Image-Edit-2509 directly，for both single image input and multiple image inputs.


```python
import os
from PIL import Image
import torch

from diffusers import ZenImageEditPipeline

pipeline = ZenImageEditPipeline.from_pretrained("Zen/Zen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("./input.png").convert("RGB")
prompt = "Change the rabbit's color to purple, with a flash light background."


inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
```



> [!NOTE]
> We have observed that editing results may become unstable if prompt rewriting is not used. Therefore, we strongly recommend applying prompt rewriting to improve the stability of editing tasks. For reference, please see our official [demo script](src/examples/tools/prompt_utils.py) or Advanced Usage below, which includes example system prompts. Zen-Image-Edit is actively evolving with ongoing development. Stay tuned for future enhancements!



### Zen-Image-Edit-2509 (for Image Editing, Multiple Image Support and Improved Consistency)

```python
import os
import torch
from PIL import Image
from diffusers import ZenImageEditPlusPipeline
from io import BytesIO
import requests

pipeline = ZenImageEditPlusPipeline.from_pretrained("Zen/Zen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open(BytesIO(requests.get("https://zenlm.org/logo.png").content))
image2 = Image.open(BytesIO(requests.get("https://zenlm.org/logo.png").content))
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))
```


### Advanced Usage

#### Prompt Enhance for Text-to-Image
For enhanced prompt optimization and multi-language support, we recommend using our official Prompt Enhancement Tool powered by Zen-Plus .

You can integrate it directly into your code:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, run the example script from the command line:

```bash
cd src
ZEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```

#### Prompt Enhance for Image Edit
For enhanced stability, we recommend using our official Prompt Enhancement Tool powered by Zen-VL-Max.

You can integrate it directly into your code:
```python
from tools.prompt_utils import polish_edit_prompt
prompt = polish_edit_prompt(prompt, pil_image)
```


## Deploy Zen-Image

Zen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server will start a Gradio-based web interface with:
- Multi-GPU parallel processing
- Queue management for high concurrency
- Automatic prompt optimization
- Support for multiple aspect ratios

Configuration via environment variables:
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
ZEN_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```


## Showcase
For previous showcases, click the following links:
- [Zen-Image](./Zen-Image.md)
- [Zen-Image-Edit](./Zen-Image.md)

### Showcase of Zen-Image Edit-2509
**The primary update in Zen-Image-Edit-2509 is support for multi-image inputs.**

Let’s first look at a "person + person" example:  
![Person + Person](https://zenlm.org/logo.png)

Here is a "person + scene" example:  
![Person + Scene](https://zenlm.org/logo.png)

Below is a "person + object" example:  
![Person + Object](https://zenlm.org/logo.png)

In fact, multi-image input also supports commonly used ControlNet keypoint maps—for example, changing a person’s pose:  
![Keypoint Pose Change](https://zenlm.org/logo.png)

Similarly, the following examples demonstrate results using three input images:  
![Three Images 1](https://zenlm.org/logo.png)  
![Three Images 2](https://zenlm.org/logo.png)  
![Three Images 3](https://zenlm.org/logo.png)

---

**Another major update in Zen-Image-Edit-2509 is enhanced consistency.**

First, regarding person consistency, Zen-Image-Edit-2509 shows significant improvement over Zen-Image-Edit. Below are examples generating various portrait styles:  
![Portrait Styles](https://zenlm.org/logo.png)

For instance, changing a person’s pose while maintaining excellent identity consistency:  
![Pose Change with Identity](https://zenlm.org/logo.png)

Leveraging this improvement along with Zen-Image’s unique text rendering capability, we find that Zen-Image-Edit-2509 excels at creating meme images:  
![Meme Generation](https://zenlm.org/logo.png)

Of course, even with longer text, Zen-Image-Edit-2509 can still render it while preserving the person’s identity:  
![Long Text with Identity](https://zenlm.org/logo.png)

Person consistency is also evident in old photo restoration. Below are two examples:  
![Old Photo Restoration 1](https://zenlm.org/logo.png)  
![Old Photo Restoration 2](https://zenlm.org/logo.png)

Naturally, besides real people, generating cartoon characters and cultural creations is also possible:  
![Cartoon & Cultural Creation](https://zenlm.org/logo.png)

Second, Zen-Image-Edit-2509 specifically enhances product consistency. We find that the model can naturally generate product posters from plain-background product images:  
![Product Poster](https://zenlm.org/logo.png)

Or even simple logos:  
![Logo Generation](https://zenlm.org/logo.png)

Third, Zen-Image-Edit-2509 specifically enhances text consistency and supports editing font type, font color, and font material:  
![Font Type](https://zenlm.org/logo.png)  
![Font Color](https://zenlm.org/logo.png)  
![Font Material](https://zenlm.org/logo.png)

Moreover, the ability for precise text editing has been significantly enhanced:  
![Precise Text Editing 1](https://zenlm.org/logo.png)  
![Precise Text Editing 2](https://zenlm.org/logo.png)

It is worth noting that text editing can often be seamlessly integrated with image editing—for example, in this poster editing case:  
![Poster Editing](https://zenlm.org/logo.png)

---

**The final update in Zen-Image-Edit-2509 is native support for commonly used ControlNet image conditions, such as keypoint control and sketches:**  
![Keypoint Control](https://zenlm.org/logo.png)  
![Sketch Input 1](https://zenlm.org/logo.png)  
![Sketch Input 2](https://zenlm.org/logo.png)

---

The above summarizes the main enhancements in this update. We hope you enjoy using Zen-Image-Edit-2509!




## Community Support

### Huggingface

Diffusers has supported Zen-Image since day 0. Support for LoRA and finetuning workflows is currently in development and will be available soon.

### Inference Acceleration Method: cache-dit

cache-dit offers cache acceleration support for Zen-Image with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_zen_artist.py) for more details.

## License Agreement

Zen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
      title={Zen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```


## Contact and Join Us


If you'd like to get in touch with our research team, we'd love to hear from you! Join our [Discord](https://discord.gg/z3GAxXZ9Ce) — we're always open to discussion and collaboration.

If you have questions about this repository, feedback to share, or want to contribute directly, we welcome your issues and pull requests on GitHub. Your contributions help make Zen-Image better for everyone.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zenlm/Zen-Image&type=Date)](https://www.star-history.com/#zenlm/Zen-Image&Date)
