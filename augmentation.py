from PIL import Image
import os
import random
import numpy as np

def rotar_imagen(ruta_imagen, grados, ruta_destino):
    # Abrir la imagen
    imagen = Image.open(ruta_imagen)

    # Rotar la imagen
    imagen_rotada = imagen.rotate(grados, expand=True)

    # Guardar la imagen rotada en un nuevo archivo
    imagen_rotada.save(ruta_destino)


img_dir = 'gesto3'
img_dir2 = 'gesto3-aug'

imgs = os.listdir(img_dir)
count = 1
for im in imgs:
	if im.lower().endswith(('.png', '.jpg', '.jpeg')):
		complete_route = os.path.join(img_dir, im)
		if(np.random.uniform(0,1) < 0.5):
			rotar_imagen(complete_route, 90, img_dir2 + "/" + str(count) + ".png")
		else:
			rotar_imagen(complete_route, -90, img_dir2 + "/" + str(count) + ".png")
		count += 1

