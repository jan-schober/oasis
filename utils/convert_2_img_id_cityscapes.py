import glob
from collections import namedtuple
from shutil import copyfile
import cv2
import numpy as np


def main():
    root_path = '/home/schober/carla/output/bdd_1280_720/'
    output_folder_rendered = '/home/schober/carla/output/for_city_converted/rendered/'
    output_folder = '/home/schober/carla/output/for_city_converted/labels/'
    img_list = glob.glob(root_path + '*semsec.png')
    img_list_rendered = glob.glob(root_path + '*cam.png')

    for source_path in img_list_rendered:
        img_number = get_carla_number(source_path)
        output_dst = output_folder_rendered + 'carla_' + str(img_number) + '_leftImg8bit.png'
        print(output_dst)
        copyfile(source_path, output_dst)

    for img in img_list:

        color_img = cv2.imread(img)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        Label = namedtuple('Label', [

            'name',  # The identifier of this label, e.g. 'car', 'person', ... .
            # We use them to uniquely name a class

            'id',  # An integer ID that is associated with this label.
            # The IDs are used to represent the label in ground truth images
            # An ID of -1 means that this label does not have an ID and thus
            # is ignored when creating ground truth images (e.g. license plate).
            # Do not modify these IDs, since exactly these IDs are expected by the
            # evaluation server.

            'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
            # ground truth images with train IDs, using the tools provided in the
            # 'preparation' folder. However, make sure to validate or submit results
            # to our evaluation server using the regular IDs above!
            # For trainIds, multiple labels might have the same ID. Then, these labels
            # are mapped to the same class in the ground truth images. For the inverse
            # mapping, we use the label that is defined first in the list below.
            # For example, mapping all void-type classes to the same ID in training,
            # might make sense for some approaches.
            # Max value is 255!

            'category',  # The name of the category that this label belongs to

            'categoryId',  # The ID of this category. Used to create ground truth images
            # on category level.

            'hasInstances',  # Whether this label distinguishes between single instances or not

            'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
            # during evaluations or not

            'color',  # The color of this label

            'instance_id'
        ])

        labels = [
            #       name     id  trainId category catId hasInstances ignoreInEval color instance_id
            Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0), -1),
            Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0), -1),
            Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0), -1),
            Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0), -1),
            Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0), -1),
            Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0), -1),
            Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81), -1),
            Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128), -1),
            Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232), -1),
            Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160), -1),
            Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140), -1),
            Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70), -1),
            Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156), -1),
            Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153), -1),
            Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180), -1),
            Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100), -1),
            Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90), -1),
            Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153), -1),
            Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153), -1),
            Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30), -1),
            Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0), -1),
            Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35), -1),
            Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152), -1),
            Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180), -1),
            Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60), 93),
            Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0), 97),
            Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142), 101),
            Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70), 105),
            Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100), 109),
            Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90), 113),
            Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110), 117),
            Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100), 121),
            Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230), 125),
            Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32), 128),
            # Label('license plate',-1, -1 ,'vehicle', 7, False,True,(0, 0,142),-1),
            Label('LANE', 7, 18, 'void', 7, False, False, (157, 234, 50), -1),
            Label('DYNAMIK', 5, 18, 'void', 7, False, False, (170, 120, 50), -1),
            Label('STATIK', 4, 18, 'void', 7, False, False, (110, 190, 160), -1),
            Label('TERRAIN', 22, 18, 'void', 7, False, False, (145, 170, 100), -1),
            Label('FENCE', 13, 18, 'void', 7, False, False, (100, 40, 40), -1),
            Label('OTHER', 0, 18, 'void', 7, False, False, (55, 90, 80), -1),
            Label('WATER', 22, 18, 'void', 7, False, False, (45, 60, 150), -1),
            Label('unlabeled', 0, 18, 'void', 7, False, False, (0, 0, 0), -1),

        ]

        img_mask = np.copy(color_img)
        color_mask = np.copy(color_img)
        h, w, c = color_img.shape
        instance_mask = np.zeros((h, w, 3), dtype='uint8')
        list_carla = ['LANE', 'DYNAMIK', 'STATIK', 'TERRAIN', 'FENCE', 'OTHER', 'WATER']
        for label in labels:
            color_rgb = label.color
            c_id = label.id
            color_id = [c_id, c_id, c_id]

            mask = cv2.inRange(color_img, color_rgb, color_rgb)
            img_mask = np.copy(img_mask)
            img_mask[mask != 0] = color_id

            instance = label.hasInstances
            if instance and mask.any() > 0:
                i_id = label.instance_id
                print(i_id)
                color_instance = [i_id, i_id, i_id]
                instance_mask = np.copy(instance_mask)
                instance_mask[mask != 0] = color_instance
            name = label.name
            if name in list_carla:
                if name == 'LANE':
                    color_sem = [128, 64, 128]
                elif name == 'DYNAMIK':
                    color_sem = [111, 74, 0]
                elif name == 'STATIK':
                    color_sem = [0, 0, 0]
                elif name == 'TERRAIN':
                    color_sem = [152, 251, 152]
                elif name == 'FENCE':
                    color_sem = [100, 40, 40]
                elif name == 'OTHER':
                    color_sem = [0, 0, 0]
                elif name == 'WATER':
                    color_sem = [152, 251, 152]
                color_mask = np.copy(color_mask)
                color_mask[mask != 0] = color_sem

        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
        img_mask = img_mask.astype('uint8')

        instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_RGB2GRAY)
        instance_mask = instance_mask.astype('uint8')

        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        color_mask = color_mask.astype('uint8')

        carla_number = get_carla_number(img)

        cv2.imwrite(output_folder + 'carla_' + str(carla_number) + '_gtFine_instanceIds.png', instance_mask)
        cv2.imwrite(output_folder + 'carla_' + str(carla_number) + '_gtFine_labelIds.png', img_mask)
        cv2.imwrite(output_folder + 'carla_' + str(carla_number) + '_gtFine_color.png', color_mask)


def get_carla_number(path):
    file_name = path.split('/')[-1]
    number = file_name.split('_')[0]
    return number


if __name__ == "__main__":
    main()
