from shutil import copyfile
import glob

root_path = '/home/schober/vkitti_conv/source/'

file_list = glob.glob(root_path + '*/*png')
dst_path = '/home/schober/vkitti_conv/images/'
for file in file_list:
    file_name = file.split('/')[-1]
    file_name= file_name.replace()
    scene = file.split('/')[5]
    destination =dst_path + scene +'_' + file_name
    copyfile(file, destination)
