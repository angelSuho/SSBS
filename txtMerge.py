import cv2
import os
import numpy as np
import time

pathT = "/Users/jhkimMultiGpus/Desktop/scene-1/Original/obj/"
# pathT = "/Users/jhkimMultiGpus/Desktop/scene-2/Original/obj/"

pwd = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData"
save_path = "/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SmokeSimulationbyCNN/enhance/outputs/SmokeTestData/MergeData"


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


def txtRead(asd):
    fileInput = open(asd, 'r')
    u = fileInput.readline()
    line = u.split()
    image = create_blank(int(line[0]), int(line[1]), (0, 0, 0))
    image = np.array(image)
    while True:
        line = fileInput.readline()
        if not line:
            break
        line = line.split()
        x = int(line[0])  # col   width
        y = int(line[1])  # row   height
        image[y][x] = (float(line[2]), float(line[3]), float(line[4]))
    return u, image


start = time.time()  # start time save
print("START")
num = 0
result_size = 1024

if not os.path.isdir(save_path):
    os.mkdir(pwd + "/MergeData")

for path, dirs, files in os.walk(pathT):
    for file in files:
        num += 1

print("count : " + str(num))

for i in range(num):
    pwd2 = pwd + "/smoke_" + str(i)
    txt = create_blank(result_size, result_size, (0,0,0))
    txt = np.array(txt)

    for path, dirs, files in os.walk(pwd2):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.txt':
                asd = pwd2 + "/" + file
                index, image = txtRead(asd)
                index = index.split()
                name = os.path.splitext(file)

                # 0:xsize 1:ysize 2:depth 3:xposition 4:yposition 5:state 6:xsize 7:ysize
                # 128 128 3 512 192 FD 128 128

                # redLine code one
                # for x in range(int(index[0])):
                #     for y in range(int(index[1])):
                #             if (y < int(index[1]) and x == 0) or (y < int(index[1]) and x == int(index[0]) - 1):  # 왼쪽세로줄
                #                 image[x, y, :] = [0, 0, 1]
                #             if (y % int(index[1])) == 0 or (y % int(index[1])) == int(index[1]) - 1:
                #                 image[x, y, :] = [0, 0, 1]

                # redLine code 1
                # for x in range(int(index[0])):
                #     for y in range(int(index[1])):
                #             if y < int(index[1]) and x == 0:
                #                 image[x, y, :] = [0, 0, 1]
                #             if (y % int(index[1])) == 0 or (y % int(index[1])) == int(index[1]) - 1:
                #                 image[x, y, :] = [0, 0, 1]
                #             if y < int(index[1]) and x == int(index[0]) - 1:
                #                 image[x, y, :] = [0, 0, 1]

                # redLine code 2
                # for x in range(int(index[0])):
                #     for y in range(int(index[1])):
                #             if (y < int(index[1]) and x == 0) or (y < int(index[1]) and x == int(index[0]) - 1):
                #                 image[x, y, :] = [0, 0, 1]
                #             if (y % int(index[1])) == 0 or (y % int(index[1])) == int(index[1]) - 1:
                #                 image[x, y, :] = [0, 0, 1]

                image *= 255.0
                x = int(index[3]) - int(index[0])          # row
                y = int(index[4]) - int(index[1])          # col
                txt[x:x+int(index[0]), y:y+int(index[1]), :] = image
    if i < 10:
        cv2.imwrite(os.path.join(save_path, "smoke_00" + str(i) + '.jpg'), txt)
    elif i < 100:
        cv2.imwrite(os.path.join(save_path, "smoke_0" + str(i) + '.jpg'), txt)
    else:
        cv2.imwrite(os.path.join(save_path, "smoke_" + str(i) + '.jpg'), txt)
    print("smoke_" + str(i) + '.jpg')
print("END")
print("time :", time.time() - start)

