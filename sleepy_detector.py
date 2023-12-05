import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import scipy.io.wavfile
import tempfile
import os
from playsound import playsound
import threading
import time
import sounddevice as sd

def f(t, f_c, f_m, beta):
    return np.sin(2 * np.pi * f_c * t - beta * np.sin(2 * f_m * np.pi * t))

def to_integer(signal):
    # Take samples in [-1, 1] and scale to 16-bit integers,
    # values between -2^15 and 2^15 - 1.
    return np.int16(signal * (2**15 - 1))

def generate_and_play_siren(duration=5, f_c=1500, f_m=2, beta=100):
    N = 48000  # samples per second
    x = np.arange(duration * N)  # duration seconds of audio

    data = f(x/N, f_c, f_m, beta)
    
    # Play the sound
    sd.play(data, samplerate=N)
    sd.wait()


# Global variable to control the siren state
siren_on = False

def start_siren():
    global siren_on
    siren_on = True
    print("Siren turned ON.")

def stop_siren():
    global siren_on
    siren_on = False
    print("Siren turned OFF.")

def play_siren():
    global siren_on
    while True:
        if siren_on:
            generate_and_play_siren(duration=1)  # Play the siren for 1 second at a time
        else:
            time.sleep(1)  # Wait for 1 second before checking the siren state again

# Start the siren player in a separate thread
siren_thread = threading.Thread(target=play_siren, daemon=True)
siren_thread.start()


def eye_asspec_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "eye_predictor.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(0, 6):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 5:
                next_point = 0
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(6, 12):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 11:
                next_point = 6
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = eye_asspec_ratio(leftEye)
        right_ear = eye_asspec_ratio(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR, 2)
        if EAR < 0.2:
            start_siren()
            cv2.putText(frame, "Are you Sleepy?", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Drowsy")
        else:
            stop_siren()
        print(EAR)

    cv2.imshow("Sleepy Detector", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
