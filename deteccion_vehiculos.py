import cv2
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class CarDetector:
    def __init__(self, cascade_path):
        self.car_classifier = cv2.CascadeClassifier(cascade_path)
    
    def detection(self, frame):
        vehicle = self.car_classifier.detectMultiScale(frame, 1.15, 4)
        for (x, y, w, h) in vehicle:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            cv2.putText(frame, 'Carro Detectado!', (x+w, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return frame
    
    def detect_in_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir el archivo de video en la ruta {video_path}")
            return

        while cap.isOpened():
            time.sleep(0.05)
            ret, frame = cap.read()
            if not ret:
                messagebox.showinfo("Información", "No se puede leer el frame del video o el video ha terminado")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = self.car_classifier.detectMultiScale(gray, 1.4, 2)
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.imshow('Cars', frame)
            if cv2.waitKey(1) == 13: # 13 es la tecla Enter
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def detect_in_realtime(self):
        realtimevideo = cv2.VideoCapture(0)
        if not realtimevideo.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            return

        while realtimevideo.isOpened():
            ret, frame = realtimevideo.read()
            if not ret:
                messagebox.showerror("Error", "No se puede leer el frame de la cámara.")
                break
            frame = self.detection(frame)
            cv2.imshow('Car Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        realtimevideo.release()
        cv2.destroyAllWindows()


class App:
    def __init__(self, root, detector):
        self.detector = detector
        self.root = root
        self.root.title("Detección de Coches")
        self.root.geometry("300x150")
        self.create_widgets()
    
    def create_widgets(self):
        video_button = tk.Button(self.root, text="Detectar en Video", command=self.select_video)
        video_button.pack(pady=10)

        realtime_button = tk.Button(self.root, text="Detectar en Tiempo Real", command=self.detector.detect_in_realtime)
        realtime_button.pack(pady=10)
    
    def select_video(self):
        video_path = filedialog.askopenfilename(title="Seleccionar archivo de video",
                                                filetypes=[("Archivos de video", "*.avi *.mp4 *.mov")])
        if video_path:
            self.detector.detect_in_video(video_path)


if __name__ == "__main__":
    cascade_path = os.path.abspath('Proyecto/haarcascade_car.xml')
    detector = CarDetector(cascade_path)

    root = tk.Tk()
    app = App(root, detector)
    root.mainloop()
