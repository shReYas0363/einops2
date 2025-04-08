# 🧠 Minimal Einops Implementation 

This is a minimalistic reimplementation of the core functionality of [`einops`](https://github.com/arogozhnikov/einops). Supports only numpy.
Initially wanted to do a Rust parser and python frontend (for the speed, but failed terribly).
einops2.py is the main module - which has a input reader and has the rearrange function.
Recipe_converter.py - CHecks input and output sides for a valid expression and converts into the numpy actions (algorithm in the code)


## 📂 Installation & Usage

1. Clone or download the files into any directory of your choice.
2. test.ipynb has the basic examples and stuff


## 🧪 Example: Vision Transformer-style Patch Extraction (with Numpy)

You can use this module similarly to `einops.rearrange()` in PyTorch-based models like Vision Transformers (ViT). Here's how you would convert an image into non-overlapping 16×16 patches:

```python
import numpy
from einops2 import rearrange  #if not in working directory mention einops2 path

#Sample image (let)
image = numpy.random.rand(1, 3, 224, 224)

# Rearranging 
patches = rearrange(
    image,
    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
    p1=16, p2=16
)

print(patches.shape)  # Output: torch.Size([1, 196, 768])
