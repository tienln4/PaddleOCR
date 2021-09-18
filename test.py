# from benchmark.main import Benchmark
# from benchmark.modules.configs.paddle_recognizer_config import PaddleRecognizerConfig
# from benchmark.modules.recognizer.paddle_recognizer import PaddleRecognizer

# config = PaddleRecognizerConfig("benchmark/configs/plate.yaml")
# bm = Benchmark(config)
# bm.ExportResult("benchmark/results/v2.txt")
# bm.GenResults()
import os
from shutil import copyfile

img_dir = "benchmark/v3/images"
save_img_dir = "DAT/images"
for fn in os.listdir(img_dir):
    fp = os.path.join(img_dir, fn)
    copyfile(os.path.join(img_dir, fn), os.path.join(save_img_dir, fn))