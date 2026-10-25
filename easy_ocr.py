import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import configparser

my_config = configparser.ConfigParser()
my_config.read("./config.ini")

reader = easyocr.Reader(['ru', 'en'], gpu=True)
img_path = my_config["PATH"]["path_to_test_image"]

img = cv2.imread(img_path)

results = reader.readtext(img_path)

img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img_pil)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except IOError:
    print("There is not font use chose")
    fong = ImageFont.load_default()

for bbox, text, prob in results:

    tl = (int(bbox[0][0]), int(bbox[0][1]))
    tr = (int(bbox[1][0]), int(bbox[1][1]))
    br = (int(bbox[2][0]), int(bbox[2][1]))
    bl = (int(bbox[3][0]), int(bbox[3][1]))

    draw.polygon([tl, tr, br, bl], outline=(0, 255, 0), width=2)

    text_to_draw = f"{text} ({prob:.2f})"
    draw.text((tl[0], tl[1] - 25), text_to_draw, font=font, fill=(255, 0, 0))

for idx, (bbox, text, prob) in enumerate(results):
    print(f"[{idx + 1}] text: {text:<25}, confidence: {prob:.3f}")

final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("result", final_img)

cv2.imwrite("./output/easy_ocr_out/easyocr_output.png", final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()