import numpy as np
import cv2
from matplotlib import pyplot as plt
original = cv2.imread('img/mano.png')
img = original.copy()
img = cv2.pyrDown(img)
mask = np.zeros(img.shape[:2], np.uint8)		

#modelo de fondo y primer plano relleno de 0
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

height, width = img.shape[:2]

# Calculate the center of the image
center_y, center_x = height // 2, width // 2

#rectangulo inicial para los modelos de grabcut
rect = (center_x-100, center_y-100, center_x+100, center_y+100) 

#ejecutar el algoritmo con lo definido anteriormente
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT) 


mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8') 
img = img*mask2[:,:,np.newaxis] 

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([])
plt.yticks([])
plt.show()
