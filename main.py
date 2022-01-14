from os import listdir
import cv2
import matplotlib.pyplot as plt
import torch
from detecto.utils import read_image
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto.core import DataLoader, Model
from detecto.visualize import show_labeled_image
import os


location_of_data = r'C:\Users\kapsi\Desktop\WdPO_P\train'
test_data = r'C:\Users\kapsi\Desktop\WdPO_P\test'
test_data = test_data + '\\'

dirs = listdir(location_of_data)
for file in dirs:
   print(file)

dirs = listdir(location_of_data)
for file in dirs:
   print(file)

base_path = location_of_data + '\\'
print(base_path)
sample_image = "s1.jpg"

#image = detecto.utils(base_path+sample_image)
# image = cv2.imread(base_path+sample_image)
# cv2.imshow('img',image)
# cv2.waitKey(0)
img = cv2.imread(base_path+sample_image)
# image = read_image(base_path+sample_image)
# plt.imshow(image)
# plt.show()
# cv2.imshow('img',img)
# cv2.waitKey(0)

dataset = Dataset(base_path)
img, targets = dataset[10]
show_labeled_image(img, targets['boxes'], targets['labels'])

labels = ['spiderman', 'venom']
model = Model(labels)
model.fit(dataset)
torch.save(model, 'model.pth')

# directory = r'C:\Users\kapsi\Desktop\WdPO_P\test'
# file_count = sum(len(files) for _, _, files in os.walk(directory))
# if file_count == 0:
#    print('Theres nothing to show! Closing...')
#
# else:
#    pics = list()
#    for i in range(file_count):
#       path = test_data + '{}.jpg'.format(i)
#       img_test = cv2.imread(path)
#       img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
#       pics.append(img_test)
#       i += 1
#    counter = 0
#    cv2.namedWindow('example')
#    while True:
#
#       key_code = cv2.waitKey(1)
#
#       if key_code % 256 == 9:
#          print("Tab hit, closing...")
#          break
#       if key_code == ord('e'):
#          counter += 1
#       if key_code == ord('q'):
#          counter -= 1
#       if counter == file_count:
#          counter = 0
#       if counter == -1:
#          counter = file_count - 1
#
#       labels, boxes, scores = model.predict(img_test)
#       print("labels", labels)
#       print("boxes", boxes)
#       print("scores", scores)
#       show_labeled_image(img_test, boxes[0], labels[0])
#    cv2.destroyAllWindows()
#
#
#
#
