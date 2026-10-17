import easyocr

reader = easyocr.Reader(['ru', 'en'], gpu=True)
img_path = "./image_for_test/image.png"

print("Start detecting text on image")

results = reader.readtext(img_path)

print("Results:")
for idx, (bbox, text, prob) in enumerate(results):
    print(f"[{idx + 1}], {text:<25}, confidence: {prob:.3f}")