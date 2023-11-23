import cv2

# Inicializar la webcam
cap = cv2.VideoCapture(2)  # El argumento '0' indica que queremos usar la cámara predeterminada

# Verificar si la webcam se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la webcam.")
    exit()

# Capturar un solo fotograma
ret, frame = cap.read()

# Verificar si la captura fue exitosa
if not ret:
    print("Error: No se pudo capturar la imagen.")
    exit()

# Guardar la imagen capturada en un archivo (puedes cambiar el nombre y formato según tus preferencias)
cv2.imwrite("img/foto2.jpg", frame)

# Liberar la webcam y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

