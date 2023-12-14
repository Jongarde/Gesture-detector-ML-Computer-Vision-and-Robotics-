import cv2
import os
import time

# Crear la carpeta 'gesto1' si no existe
output_folder = 'gesto2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializar la cámara (cambiar 0 por el índice de la cámara si se utiliza una cámara externa)
cap = cv2.VideoCapture(2)

# Configurar el nombre del archivo y el formato de la imagen
file_prefix = 'img'
file_format = '.png'

# Configurar la frecuencia de captura en segundos y el límite de capturas
captura_cada_segundo = 20
limite_capturas = 3000

# Inicializar contador de capturas
contador_capturas = 2000

try:
    while contador_capturas < limite_capturas:
        # Capturar frame
        ret, frame = cap.read()

        if not ret:
            print("Error al capturar el frame.")
            break

        # Mostrar la imagen en directo
        cv2.imshow('Captura en Vivo', frame)
        cv2.waitKey(1)  # Necesario para que OpenCV actualice la ventana

        # Generar nombre de archivo
        file_name = f'{file_prefix}_{contador_capturas + 1:03d}{file_format}'

        # Guardar la captura en la carpeta 'gesto1'
        cv2.imwrite(os.path.join(output_folder, file_name), frame)

        print(f'Captura {contador_capturas + 1}/{limite_capturas} guardada como {file_name}')

        # Incrementar el contador de capturas
        contador_capturas += 1

        # Esperar el tiempo especificado antes de la próxima captura
        time.sleep(1/captura_cada_segundo)

finally:
    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

