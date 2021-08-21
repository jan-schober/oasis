from PIL import Image
import glob

'''
Small script to create image grids for visualisation.
'''

def image_grid(imgs, rows, cols):
    print(rows * cols)
    assert len(imgs) == rows *cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size= (cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

list_rend = sorted(glob.glob('/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd100_sunny/_auswahl/rendered/*.jpg'))
list_lab = sorted(glob.glob('/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd100_sunny/_auswahl/labels/*.png'))
list_aug = sorted(glob.glob('/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd100_sunny/_auswahl/images/*.png'))

print(len(list_rend))
print(len(list_lab))
print(len(list_aug))
final_list = []

for file in list_rend:
    final_list.append(file)
for file in list_lab:
    final_list.append(file)
for file in list_aug:
    final_list.append(file)

image, *images = [Image.open(file) for file in final_list]

grid = image_grid([image, *images], rows=3, cols = 4)
#grid.show()

grid.save('/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd100_sunny/_auswahl/oasis_grid.png')
