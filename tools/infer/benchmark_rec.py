from tools.infer.predict_rec import TextRecognizer
import typing


class PaddleRecognizer(object):
    def __init__(self, config):
        self.config = config
        self.text_recognizer = TextRecognizer(self.config)

    def recognize(self, img_list: list):
        if not isinstance(img_list, list):
            img_list = [img_list]

        ret, et = self.text_recognizer(img_list)
        return ret
