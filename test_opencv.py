import cv2, os

img_path = r"C:\Users\isha1\Melodora\test_image_converted.jpg"

print("cv2 version:", cv2.__version__)
print("Image exists:", os.path.exists(img_path))

img = cv2.imread(img_path)
print("cv2.imread result:", type(img))

if img is None:
    print("❌ OpenCV failed to load the image.")
else:
    print("✅ Image loaded successfully. Shape:", img.shape)
