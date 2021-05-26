from PIL import Image
import glob

'''
Small script to create image grids for visualisation.
'''

def image_grid(imgs, rows, cols):
    print(len(imgs))
    assert len(imgs) == rows *cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size= (cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

root_path = '/media/jan/TEST/_Masterarbeit/results/vergleich_cut_star/'
path_carla_output = '/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd_result/carla_resized/'
list_cut = sorted(glob.glob(root_path + 'cut/*.png'))
#list_renderd = sorted(glob.glob(root_path + 'rendered/*.jpg'))
list_star = sorted(glob.glob(root_path + 'star/*.jpg'))
#list_synth_old = sorted(glob.glob('/media/jan/TEST/_Masterarbeit/results/results_OASIS/bdd100knew/best_137500/image/*png'))
#list_synth_tue_1 = sorted(glob.glob(root_path + 'best_carla_tuesday_1/image/*.png'))
#list_synth_city = sorted(glob.glob(root_path + 'city_vs_bdd/image/*.png'))
#list_label = sorted(glob.glob(root_path + 'latest_150k/label/*.png'))
list_rend = sorted(glob.glob(root_path + 'rendered/*.png'))

final_list = []

for file in list_rend:
    final_list.append(file)
for file in list_cut:
    final_list.append(file)
for file in list_star:
    final_list.append(file)


print(len(final_list))
image, *images = [Image.open(file) for file in final_list]

grid = image_grid([image, *images], rows=3, cols = 11)
#grid.show()

grid.save('/media/jan/TEST/_Masterarbeit/results/vergleich_cut_star/grid_cut_star_vergleich.png')
