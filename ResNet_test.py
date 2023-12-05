import cv2
import numpy as np
from math import *
import random
import torch
import torch.nn as nn
from torchvision import models
from scipy.spatial import distance
import dlib


class ResNet50(nn.Module):
    def __init__(self,num_classes=24):
        super().__init__()
        self.model_name='resnet50'
        self.model=models.resnet50()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x
    
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio
    
model = ResNet50()
model.load_state_dict(torch.load('weights\model_weights.pth'))
model.eval()
if torch.cuda.is_available():
    model.cuda()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add an extra batch dimension
    if torch.cuda.is_available():
        img = img.cuda()
    return img

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()  # Initialize the dlib face detector

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]

        input_tensor = preprocess_image(face_region)
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = ((predictions.cpu().numpy().reshape(-1, 12, 2) + 0.5) * w).squeeze()  # Adjusted to 12 landmarks

        for (px, py) in predictions:
            cv2.circle(frame, (int(px) + x, int(py) + y), 1, (0, 255, 0), 1)

    cv2.imshow("Eyes Landmarks", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
