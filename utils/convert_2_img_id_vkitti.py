import cv2
import glob
import numpy as np
from collections import namedtuple
from shutil import copyfile

'''
This skript convertes the vkitti semantic images to bdd semantic image format.
'''

def main():
    root_path = '/storage/VKitti/vkitti/'
    output_folder_rendered = '/home/schober/vkitti_conv/rendered/'
    output_folder = '/home/schober/vkitti_conv/labels/'
    img_list = glob.glob(root_path + '*/clone/frames/classSegmentation/Camera_0/*.png')
    img_list_rendered = glob.glob(root_path +  '*/clone/frames/rgb/Camera_0/*.jpg')

    print(img_list_rendered[0])
    for source_path in img_list_rendered:
        file_name = source_path.split('/')[-1]
        file_name = file_name.replace('rgb', '')
        file_name = file_name.replace('png', 'jpg')
        scene = source_path.split('/')[4]
        out_name = scene+file_name
        renderd_image = cv2.imread(source_path)
        renderd_image = cv2.resize(renderd_image, (512, 256))
        output_dst = output_folder_rendered + out_name
        cv2.imwrite(output_dst, renderd_image)

    Label = namedtuple('Label', [
        'name',
        'id',
        'color',
    ])
    # 255 == ignore
    labels = [
        #      name   id  color
        Label('Terrain', 9, (210, 0, 200)),
        Label('Sky', 10, (90, 200, 255)),
        Label('Tree', 8, (0, 199, 0)),
        Label('Vegetation', 8, (90, 240, 0)),
        Label('Building', 2, (140, 140, 140)),
        Label('Road', 0, (100, 60, 100)),
        Label('GuardRail', 255, (250, 100, 255)),
        Label('TrafficSign', 7, (255, 255, 0)),
        Label('TrafficLight', 6, (200, 200, 0)),
        Label('Pole', 5, (255, 130, 0)),
        Label('Misc', 255, (80, 80, 80)),
        Label('Truck', 14, (160, 60, 60)),
        Label('Car', 13, (255, 127, 80)),
        Label('Van', 13, (0, 139, 139)),
        Label('Undefined', 255, (0, 0, 0)),
    ]

    for img in img_list:
        color_img = cv2.imread(img)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img,(512,256),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
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

        file_name = img.split('/')[-1]
        file_name = file_name.replace('classgt', '')
        scene = img.split('/')[4]
        out_name = scene + file_name

        cv2.imwrite(output_folder + out_name, img_mask)

if __name__ == "__main__":
    main()
