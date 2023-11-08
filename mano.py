import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe_functions as mf

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path='model/gesture_recognizer.task'

mp_image = mp.Image.create_from_file('mano.png')

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:
	gesture_recognition_result = recognizer.recognize(mp_image)
	top_gesture = gesture_recognition_result.gestures[0][0]
	hand_landmarks = gesture_recognition_result.hand_landmarks

images = []
results = []
images.append(mp_image)
results.append((top_gesture, hand_landmarks))	
mf.display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
