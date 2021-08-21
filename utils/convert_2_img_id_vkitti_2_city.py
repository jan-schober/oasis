import glob
from collections import namedtuple
from shutil import copyfile
import cv2
import numpy as np

'''
This script converts vkitti labels to cityscapes semantic label, instance id and label color.
It also copy the rendered images.
The file names are equal to the cityscape file names.
'''
def main():
    root_path = '/storage/VKitti/vkitti/'
    output_folder_rendered = '/home/schober/cityscape_dataset/leftImg8bit/val/vkitti/'
    output_folder = '/home/schober/cityscape_dataset/gtFine/val/vkitti/'

    img_list = glob.glob(root_path + '*/clone/frames/classSegmentation/Camera_0/*.png')
    img_list_rendered =glob.glob(root_path +  '*/clone/frames/rgb/Camera_0/*.jpg')

    for source_path in img_list_rendered:
        file_name = get_name(source_path)
        file_name = file_name.replace('rgb', '')
        scene = source_path.split('/')[4]
        out_name = scene + file_name + '_leftImg8bit.png'
        renderd_image = cv2.imread(source_path)
        renderd_image = cv2.resize(renderd_image, (1280, 720))
        output_dst = output_folder_rendered + out_name
        cv2.imwrite(output_dst, renderd_image)

    for img in img_list:

        color_img = cv2.imread(img)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        Label = namedtuple('Label', [
            'name',
            'id',
            'color',  # The color of this label
        ])

        labels = [
            #       name     id     color
            Label('Terrain', 22, (210, 0, 200)),
            Label('Sky', 23, (90, 200, 255)),
            Label('Tree', 21, (0, 199, 0)),
            Label('Vegetation', 21, (90, 240, 0)),
            Label('Building', 11, (140, 140, 140)),
            Label('Road', 7, (100, 60, 100)),
            Label('GuardRail', 14, (250, 100, 255)),
            Label('TrafficSign', 20, (255, 255, 0)),
            Label('TrafficLight', 19, (200, 200, 0)),
            Label('Pole', 17, (255, 130, 0)),
            Label('Misc', 0, (80, 80, 80)),
            Label('Truck', 27, (160, 60, 60)),
            Label('Car', 26, (255, 127, 80)),
            Label('Van', 26, (0, 139, 139)),
            Label('Undefined', 0, (0, 0, 0)),
        ]

        img_mask = np.copy(color_img)
        for label in labels:
            color_rgb = label.color
            c_id = label.id
            color_id = [c_id, c_id, c_id]

            mask = cv2.inRange(color_img, color_rgb, color_rgb)
            img_mask = np.copy(img_mask)
            img_mask[mask != 0] = color_id

        img_mask = cv2.resize(img_mask,(1280,720),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
        img_mask = img_mask.astype('uint8')


        file_name = get_name(img)
        file_name = file_name.replace('classgt', '')
        scene = img.split('/')[4]
        out_name =scene + file_name + '_gtFine_labelIds.png'
        cv2.imwrite(output_folder + out_name, img_mask)



def get_name(path):
    file_name = path.split('/')[-1]
    name_without_ending = file_name.split('.')[0]
    return name_without_ending


if __name__ == "__main__":
    main()
