import cv2

# Carga el modelo Haar Cascade para detección de manos
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('model/haarcascade_eye.xml')

# Crea un objeto VideoCapture para capturar el video desde la webcam (por defecto, la cámara 0)
cap = cv2.VideoCapture(0)

while True:
	# Lee un frame de la webcam
	ret, frame = cap.read()

	if not ret:
		break

	# Convierte el frame a escala de grises
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecta manos en el frame usando el clasificador Haar Cascade
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
	eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.28, minNeighbors=15)
	
	# Dibuja rectángulos alrededor de los ojos detectadas
	for (x, y, w, h) in eyes:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	# Dibuja rectángulos alrededor de las caras detectadas
	for (x, y, w, h) in faces:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	
	# Muestra el frame con la detección de manos en tiempo real
	cv2.imshow('Face Detection', frame)

	# Rompe el bucle cuando se presiona la tecla 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Libera el objeto VideoCapture y cierra todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()

