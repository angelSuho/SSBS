import cv2
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float32)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image

def real_num(data,width,height, pwd):
    file = open(pwd + '.txt', 'w')
    w_h = str(width) + " " + str(height) + "\n"
    file.write(w_h)
    for y in range(height):
        for x in range(width):
            write_rgb = "{0} {1}".format(x,y)
            rgb=" {0} {1} {2}\n".format(data[x][y][0], data[x][y][1], data[x][y][2])
            file.write(write_rgb+rgb)
    file.close()

image = create_blank(512,512,(0,0,0))

for x in range(512):
    for y in range(512):
        if (256-x)**2+(256-y)**2 <= 22500:
            image[x,y,:] = [255,255,255]

tmp_save_path = "circle"
cv2.imwrite(tmp_save_path + ".jpg", image)
real_num(image, 512, 512, tmp_save_path)
print("SUCCESS")


