import cv2
from matplotlib import pyplot as plt

# get le list of files in current directory
# (not used in this snippet, but can be useful for debugging)
# import os
# files = os.listdir('.')
# print(files)
img = cv2.imread('opencv/sample2.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("La mia immagine")
plt.axis('off')
plt.show()

