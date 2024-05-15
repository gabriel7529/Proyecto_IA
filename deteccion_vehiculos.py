import cv2
import time
import numpy as np
import os




cascade_path = os.path.abspath('Proyecto/haarcascade_car.xml')
car_classifier = cv2.CascadeClassifier(cascade_path)
video_path = os.path.abspath('Proyecto/video.avi')
cap = cv2.VideoCapture(video_path)

def detection(frame):
    vehicle = car_classifier.detectMultiScale(frame, 1.4, 2)






if not cap.isOpened():
    print(f"Error: No se pudo abrir el archivo de video en la ruta {video_path}")
    exit()
# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(0.05)
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el frame del video o el video ha terminado")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) ==13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()