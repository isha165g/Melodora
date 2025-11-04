# IMPORTS
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# LOAD MODEL AND SETUP
# -----------------------------
model = load_model("emotion_recognition_model.h5")

# Emotion labels (must match your dataset folders)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load OpenCV's pre-trained face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# EMOTION DETECTION FROM IMAGE
# -----------------------------
def detect_emotion_from_image(image_path):
    """Predict emotion from an uploaded image file."""
    frame = cv2.imread(image_path)

    # Fallback if OpenCV fails to read (HEIC, AVIF, etc.)
    if frame is None:
        try:
            img = Image.open(image_path).convert("RGB")
            frame = np.array(img)[:, :, ::-1].copy()  # Convert RGB → BGR for OpenCV
            print(f"⚠️ OpenCV couldn't read {os.path.basename(image_path)}, used Pillow instead.")
        except Exception as e:
            raise ValueError(f"❌ Failed to load image: {image_path}\nReason: {e}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("⚠️ No faces detected in the image.")
        resized_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = resized_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]
        print(f"🧠 Predicted emotion (whole image): {label}")
        return

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]
        print(f"✅ Detected emotion: {label}")

        # Draw rectangle + label
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    # Display the result
    cv2.imshow("Emotion Detection (Image)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------
# EMOTION DETECTION VIA WEBCAM
# -----------------------------
def detect_emotion_from_webcam():
    """Detect emotion in real-time from webcam feed."""
    cap = cv2.VideoCapture(0)
    print("\n🎥 Webcam started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]

            # Draw face box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Live Emotion Detection (Webcam)", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# USER MODE SELECTION
# -----------------------------
def main():
    print("\n✨ Melodora - Emotion Detection ✨")
    print("1️⃣  Detect emotion from image upload")
    print("2️⃣  Detect emotion in real-time via webcam")
    
    choice = input("\nEnter your choice (1 or 2): ")

    if choice == '1':
        image_path = input("Enter full path to your image: ").strip()
        if os.path.exists(image_path):
            detect_emotion_from_image(image_path)
        else:
            print("❌ File not found. Please check the path and try again.")
    elif choice == '2':
        detect_emotion_from_webcam()
    else:
        print("❌ Invalid choice. Please restart and enter 1 or 2.")

if __name__ == "__main__":
    main()