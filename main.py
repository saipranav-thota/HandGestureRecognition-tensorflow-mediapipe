import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('models\gestureRecognition2.h5')  # Replace with your model file path

# Define the input shape for the model
input_shape = (256, 256, 1)  # Adjust based on your model input shape

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to the model's input shape
    resized_frame = cv2.resize(gray_frame, (input_shape[1], input_shape[0]))

    # Expand dimensions to match model input shape (1, height, width, channels)
    expanded_frame = np.expand_dims(resized_frame, axis=-1)  # Add channel dimension
    expanded_frame = np.expand_dims(expanded_frame, axis=0)  # Add batch dimension

    # Normalize the frame
    normalized_frame = expanded_frame / 255.0  # Normalize to [0, 1]

    # Make predictions
    predictions = model.predict(normalized_frame)
    predicted_class = np.argmax(predictions, axis=1)

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted Class: {predicted_class[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math

# cap = cv2.VideoCapture(0)
# detect = HandDetector(maxHands=1)
# classifier = Classifier("models\gestureRecognition2.h5", "model/labels.txt")


# offset = 20
# imgSize = 300
# labels = ["Palm", "Fist", "Thumb", "Like"]




# while True:
#     success, img = cap.read()
#     hands, img = detect.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgBg = np.ones((imgSize, imgSize, 3), np.uint8)*255

#         imgCrop = img[y-offset: y + h + offset, x - offset: x + w + offset]

#         aspetRatio = h / w

#         if aspetRatio > 1:
#             k = imgSize / h
#             wcal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wcal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wcal)/2)
#             imgBg[:, wGap:wcal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(img)
#             label = labels[index]
#             print(prediction, index)
#         else:
#             k = imgSize / w
#             hcal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hcal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hcal)/2)
#             imgBg[hGap:hcal + hGap, : ] = imgResize


#         cv2.imshow("ImgCrop", imgCrop)
#         cv2.imshow("ImageBackground", imgBg)
    
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)