import os

import imageio
import numpy as np
from tqdm import tqdm

video_path = "outputs"

video = []

for name in ["glee", "sam2", "ram_plus_sam2"]:
    video_name = f"{video_path}/{name}.mp4"
    imgs = []
    # Read video
    reader = imageio.get_reader(video_name)
    for i, im in enumerate(reader):
        imgs.append(im)
    video.append(imgs)

shape = video[0][0].shape
print(shape)
writer = imageio.get_writer(f"{video_path}/combined.mp4", fps=1)
res = np.zeros((shape[0] * 2, shape[1], shape[2]), dtype=np.uint8)
for i in tqdm(range(len(video[0]))):
    res[:shape[0], :, :] = video[0][i]
    res[shape[0]:, :shape[1]//2, :] = video[1][i][:, shape[1]//2:, :]
    res[shape[0]:, shape[1]//2:, :] = video[2][i][:, shape[1]//2:, :]
    writer.append_data(res)

writer.close()