import cv2
import numpy as np
import glob

img_array = []
original_array = []
image_size = 1024
j=0
for filename in glob.glob('/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData/MergeData/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    j+=1
pwd = '/Users/jhkimMultiGpus/Desktop/scene-1/After Enhance/(32x32) x2 Residual/'
for i in range(j):
    filename = '/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData(Full Size)/'+str(i)+'.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    data_org = np.array(img, dtype="float32")
    data_org_new = np.zeros((height, width, 3), np.float32)
    for r in range(height):
        for c in range(width):
            for w in range(3):
                data_org_new[r, c, w] = data_org[(image_size-1)-r, c, w]
    data_tree = np.array(img_array[i],dtype = "float32")
    residual = (data_org_new - data_tree)**2
    tmp_save_path = pwd + str(i)+".png"
    print(i)
    cv2.imwrite(tmp_save_path, residual)
    #original_array.append(img)

