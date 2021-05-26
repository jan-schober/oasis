import cv2
import glob
import numpy as np
from collections import namedtuple
from shutil import copyfile

'''
This script converts carla semantic image to bdd semantic format.
'''


def main():
    root_path = '/home/schober/carla/output/bdd_1280_720/'
    output_folder_rendered = '/home/schober/bdd100k/images/10k/val_carla/'
    output_folder = '/home/schober/bdd100k/labels/sem_seg/masks/val_carla/'
    img_list = glob.glob(root_path + '*semsec.png')
    img_list_rendered = glob.glob(root_path + '*cam.png')

    for source_path in img_list_rendered:
        img_number = get_carla_number(source_path)
        output_dst = output_folder_rendered + 'carla_' + str(img_number) + '.jpg'
        print(output_dst)
        copyfile(source_path, output_dst)

    for img in img_list:

        color_img = cv2.imread(img)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        Label = namedtuple('Label', [
            'name',
            'id',
            'color',
        ])

        # 255 == ignore

        labels = [
            #      name   id  color
            Label('road', 0, (128, 64, 128)),
            Label('sidewalk', 1, (244, 35, 232)),
            Label('building', 2, (70, 70, 70)),
            Label('wall', 3, (102, 102, 156)),
            Label('FENCE', 4, (100, 40, 40)),
            Label('pole', 5, (153, 153, 153)),
            Label('traffic light', 6, (250, 170, 30)),
            Label('traffic sign', 7, (220, 220, 0)),
            Label('vegetation', 8, (107, 142, 35)),
            Label('TERRAIN', 9, (145, 170, 100)),
            Label('sky', 10, (70, 130, 180)),
            Label('person', 11, (220, 20, 60)),
            Label('rider', 12, (255, 0, 0)),
            Label('car', 13, (0, 0, 142)),
            Label('truck', 14, (0, 0, 70)),
            Label('bus', 15, (0, 60, 100)),
            Label('train', 16, (0, 80, 100)),
            Label('motorcycle', 17, (0, 0, 230)),
            Label('bicycle', 18, (119, 11, 32)),
            ##
            Label('LANE', 0, (157, 234, 50)),
            Label('DYNAMIK', 255, (170, 120, 50)),
            Label('STATIK', 255, (110, 190, 160)),
            Label('OTHER', 255, (55, 90, 80)),
            Label('WATER', 255, (45, 60, 150)),

            Label('Ground', 255, (81, 0, 81)),
            Label('Bridge', 255, (150, 100, 100)),
            Label('RailTrack', 255, (230, 150, 140)),
            Label('GuardRail', 255, (180, 165, 180)),
            Label('unlabeled', 255, (0, 0, 0)),
        ]

        img_mask = np.copy(color_img)

        for label in labels:
            color_rgb = label.color
            c_id = label.id
            color_id = [c_id, c_id, c_id]

            mask = cv2.inRange(color_img, color_rgb, color_rgb)
            img_mask = np.copy(img_mask)
            img_mask[mask != 0] = color_id

        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
        img_mask = img_mask.astype('uint8')

        carla_number = get_carla_number(img)
        cv2.imwrite(output_folder + 'carla_' + str(carla_number) + '.png', img_mask)

def get_carla_number(path):
    file_name = path.split('/')[-1]
    number = file_name.split('_')[0]
    return number

if __name__ == "__main__":
    main()
