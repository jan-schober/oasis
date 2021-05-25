import glob
import cv2

#root_path = '/media/jan/TEST/_Masterarbeit/results/results_star_gan/vergleich/carla/'
#output_folder = '/media/jan/TEST/_Masterarbeit/results/results_star_gan/vergleich/carla_resized/'
root_path = '/home/schober/vkitti_conv/images/'
output_folder = '/home/schober/vkitti_conv/output/'
img_list = glob.glob(root_path + '*.png')


def get_name(path):
    file_name = path.split('/')[-1]
    return file_name

i = 0
max_1 = len(img_list)
w, h = 512, 256
for file in img_list:
    print(str(i) + ' / '+ str(max_1))
    i +=1
    img = cv2.imread(file)
    img = cv2.resize(img, (w, h))
    file_name = get_name(file)
    cv2.imwrite(output_folder + file_name, img)
