# IMPORTS
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Load the trained model
model = load_model("emotion_recognition_model.h5")

# Emotion labels (must match your dataset folders)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load OpenCV's pre-trained face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect emotion from an image
def detect_emotion(image_path):
    # Try to read with OpenCV first
    frame = cv2.imread(image_path)

    # If OpenCV fails (e.g., AVIF, HEIC), use Pillow as a fallback
    if frame is None:
        try:
            img = Image.open(image_path).convert("RGB")
            frame = np.array(img)[:, :, ::-1].copy()  # Convert RGB → BGR for OpenCV
            print(f"⚠️ OpenCV couldn't read {os.path.basename(image_path)}, used Pillow instead.")
        except Exception as e:
            raise ValueError(f"❌ Failed to load image: {image_path}\nReason: {e}")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("⚠️ No faces detected in the image.")
        # Optional: still try to predict emotion from the full image
        resized_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = resized_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        print(f"🧠 Predicted emotion (whole image): {label}")
        return

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            print(f"✅ Detected emotion: {label}")

            # Draw on image
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    # Show the final image
    cv2.imshow("Emotion Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
detect_emotion(r"C:\Users\isha1\Melodora\multiple1.jpg")  # replace with your own image
