from PIL import Image
img_path = r"C:\Users\isha1\Melodora\test_image.jpg"

try:
    with Image.open(img_path) as im:
        print("✅ PIL can open it.")
        print("Format:", im.format)
        print("Mode:", im.mode)
        print("Size:", im.size)
except Exception as e:
    print("❌ PIL failed to open the image:", e)
