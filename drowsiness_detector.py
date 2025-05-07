import cv2
import pygame
import time

# Initialize Pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # Must be in same directory

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Constants
EYE_CLOSED_FRAMES_THRESHOLD = 20
eye_closed_counter = 0
alarm_on = False

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        if len(eyes) >= 1:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    if eyes_detected:
        eye_closed_counter = 0
        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False
    else:
        eye_closed_counter += 1
        if eye_closed_counter >= EYE_CLOSED_FRAMES_THRESHOLD and not alarm_on:
            pygame.mixer.music.play()
            alarm_on = True
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
