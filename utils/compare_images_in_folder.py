import os

dataroot = '/home/schober/bdd100k'

mode = 'val'
path_img = os.path.join(dataroot, "images", "10k", mode)
path_lab = os.path.join(dataroot, "labels", "sem_seg", "colormaps", mode)

images = []
for item in sorted(os.listdir(path_img)):
    image = os.path.join(path_img, item)
    images.append(image.replace(".jpg", ""))

labels = []
for item in sorted(os.listdir(path_lab)):
    label = os.path.join(path_img, item)
    labels.append(label.replace(".png", ""))

#assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
list_in_images = []
list_in_labels = []
for i in range(len(images)):
    if images[i] not in labels:
        list_in_images.append(images[i])
    if labels[i] not in images:
        list_in_labels.append(labels[i])


print(list_in_images)
print(list_in_labels)
