from PIL import Image
import numpy as np

# 生成 64x64 随机噪声
noise = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
img = Image.fromarray(noise)

# 保存为 noise.png
file_path = "noise.png"
img.save(file_path)

