import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

model = tf.keras.models.load_model('models/gestureRecognition2.h5')

cap = cv2.VideoCapture(0)
detect = HandDetector(maxHands=1)

offset = 20
imgSize = 300
labels = ["Fist", "Like", "Palm", "Thumb"]
predicted_labels = []

while True:
    success, img = cap.read()
    hands, img = detect.findHands(img, flipType=False, draw=True) 

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] if 'bbox' in hand else (0, 0, img.shape[1], img.shape[0])

        imgBg = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[max(0, y-offset): min(img.shape[0], y + h + offset),
                      max(0, x - offset): min(img.shape[1], x + w + offset)]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgBg[:, wGap:wCal + wGap] = imgResize
            gray_img = cv2.cvtColor(imgBg, cv2.COLOR_BGR2GRAY)
            prediction = model.predict(np.array([gray_img]))
            predicted_class = np.argmax(prediction, axis=1)
            predicted_labels = [labels[i] for i in predicted_class]
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgBg[hGap:hCal + hGap, :] = imgResize
            gray_img = cv2.cvtColor(imgBg, cv2.COLOR_BGR2GRAY)
            prediction = model.predict(np.array([gray_img]))
            predicted_class = np.argmax(prediction, axis=1)
            predicted_labels = [labels[i] for i in predicted_class]

        text = predicted_labels[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(predicted_labels[0], font, font_scale, thickness)
        position = (x + w - text_width, y - 35) 

        cv2.putText(img, text, position, font, font_scale, color, thickness)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
