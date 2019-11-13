import torch
import matplotlib.pyplot as plt
from PIL import Image

import cv2

print(torch.__version__)
img=Image.open('./data01/images/1.jpg')
plt.imshow(img)
plt.show()
img=img.resize((60,40))
# print(img.size)
plt.imshow(img)
plt.show()