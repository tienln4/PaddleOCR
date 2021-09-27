from tools.infer.predict_rec import TextRecognizer
from .rec_object import RecObject
import typing


class PaddleRecognizer(object):
    def __init__(self, config):
        self.config = config
        self.text_recognizer = TextRecognizer(self.config)

    def recognize(self, img_list: list) -> typing.List[RecObject]:
        if not isinstance(img_list, list):
            img_list = [img_list]

        ret, et = self.text_recognizer(img_list)
        rec_objs = [RecObject(text, score, et) for text, score in ret]
        return rec_objs
