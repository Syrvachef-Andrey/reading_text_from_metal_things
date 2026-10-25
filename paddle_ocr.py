from paddleocr import PaddleOCR
import configparser

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='ru',
    enable_mkldnn=False
)

my_config = configparser.ConfigParser()
my_config.read("./config.ini")

img_path = my_config["PATH"]["path_to_test_image"]
result = ocr.predict(img_path)

for res in result:
    res.save_to_img("./output/paddle_ocr_out/")

    for text, score in zip(res['rec_texts'], res['rec_scores']):
        print(f"{text}  (conf: {score:.3f})")