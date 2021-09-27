class RecObject:
    def __init__(self, text: str, score: float, inference_time: float):
        self.text = text
        self.score = score
        self.inference_time = inference_time

    def __str__(self) -> str:
        return "'{}' ({:.3f}) ({:.3f})".format(self.text, self.score, self.inference_time)