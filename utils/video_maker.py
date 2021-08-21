import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw

filled = sorted(glob.glob('/home/schober/carla/output_cityscapes_resized/images/val_filled/*.png'))
not_filled = sorted(glob.glob('/home/schober/carla/output_cityscapes_resized/images/val_not_filled/*.png'))

assert len(filled) ==  len(not_filled),  "Different Lenght of lists"

video_name = 'carla_filled_not_filled.avi'

#black_image = Image.new('RGB', (512, 256), (0, 0, 0))


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

first_list = [filled[0], not_filled[0]]
image, *images = [Image.open(file) for file in first_list]
example_grid = image_grid([image, *images], rows=1, cols=2)
#example_grid.save('example_grid.png')

width, height = example_grid.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for filled_path, not_filled_path in zip(filled[::5], not_filled[::5]):

    filled_img = Image.open(filled_path)
    not_filled_img = Image.open(not_filled_path)

    grid = image_grid([filled_img, not_filled_img], rows=1, cols=2)
    grid_cv = np.array(grid)
    grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
    video.write(grid_cv)
video.release()
