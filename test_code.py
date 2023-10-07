import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image

table_images = os.listdir('cropped_table_images')
print(len(table_images))
print(table_images[0])
img_PIL = Image.open('./cropped_table_images/'+table_images[0])
img = np.array(img_PIL)

plt.imshow(img)
plt.show()