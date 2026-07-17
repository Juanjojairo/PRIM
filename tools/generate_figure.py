from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets import MNIST

out_path = Path("mnist_train_grid_8x8_resized32_real.png")

ds = MNIST(root="./data", train=True, download=True)

grid_n = 8
cell = 32
gap = 3
canvas_size = grid_n * cell + (grid_n + 1) * gap

canvas = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(canvas)

labels = []

for idx in range(64):
    img, label = ds[idx]
    labels.append(label)

    img32 = img.resize((32, 32), Image.Resampling.BICUBIC)

    r, c = divmod(idx, grid_n)
    x0 = gap + c * (cell + gap)
    y0 = gap + r * (cell + gap)

    draw.rectangle([x0 - 1, y0 - 1, x0 + cell, y0 + cell], outline=70)
    canvas.paste(img32, (x0, y0))

canvas.save(out_path)

print("Saved:", out_path)
print("Labels:", labels)