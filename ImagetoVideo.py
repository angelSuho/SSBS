import cv2
import numpy as np
import glob

# pwd = '/Users/jhkimMultiGpus/Desktop/scene-1/After Enhance/x4 MergeImage/'
pwd = '/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial\SmokeSimulationbyCNN/enhance/outputs/SmokeTestData/MergeData/'

img_array = []
print("path = " + pwd)
for filename in glob.glob(pwd + '*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('32x32.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("SUCCESS")