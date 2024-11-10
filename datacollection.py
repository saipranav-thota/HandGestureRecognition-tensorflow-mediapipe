import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detect = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/fist"
counter = 110

while True:
    success, img = cap.read()
    hands, img = detect.findHands(img, flipType=False) 

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
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgBg[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("ImageBackground", imgBg)
    
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgBg)
        print("Image saved:", counter)
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
