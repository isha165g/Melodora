from PIL import Image

img_path = r"C:\Users\isha1\Melodora\test_image.jpg"
new_path = r"C:\Users\isha1\Melodora\test_image_converted.jpg"

Image.open(img_path).convert("RGB").save(new_path, "JPEG")
print("✅ Saved a clean JPEG:", new_path)
