import json
import glob
path_to_json = '/home/schober/bdd100k/labels/bdd100k_labels_images_val.json'
root_path_val_images = '/home/schober/bdd100k/images/10k/val/'

val_image_list = sorted(glob.glob(root_path_val_images + '*jpg'))
val_images = []
for image in val_image_list:
    val_images.append(image.split('/')[-1])

for image in val_image_list:
    image = image.split('/')[-1]
    val_images.append(image.replace(".png", ".jpg"))

# weather and time of day we dont want in the dataset
weather_list = ['rainy', 'snowy', 'foggy']


bad_weather = []
night_images = []
weather_undefined = []
tod_undefined = []

with open(path_to_json, 'r') as f:
    labels = json.load(f)
counter = 0
for x in range(0, len(labels)):
    if labels[x]['name'] in val_images:
        counter += 1
        if labels[x]['attributes']['weather'] in weather_list:
            bad_weather.append(labels[x]['name'])
        elif labels[x]['attributes']['timeofday'] == 'night':
            night_images.append(labels[x]['name'])
        elif labels[x]['attributes']['weather'] == 'undefined':
            weather_undefined.append(labels[x]['name'])
        elif labels[x]['attributes']['timeofday'] == 'undefined':
            tod_undefined.append(labels[x]['name'])

print(len(val_images))
print(val_images[0])
print('Total number val images {}'.format(counter))
print('Number bad weather images {}'.format(len(bad_weather)))
print('Number night images {}'.format(len(night_images)))
print('Number undefined weather {}'.format(len(weather_undefined)))
print('Number undefined time of day {}'.format(len(tod_undefined)))

