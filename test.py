from benchmark.main import Benchmark
from benchmark.modules.configs.paddle_recognizer_config import PaddleRecognizerConfig
from benchmark.modules.recognizer.paddle_recognizer import PaddleRecognizer

config = PaddleRecognizerConfig("benchmark/configs/plate.yaml")
bm = Benchmark(config)
# bm.GenPythonRecord()
# bm.GenCppRecord("benchmark/results/v2_tensorrt.txt", "tensorrt")
bm.CalculateAccuracy("benchmark/results/v2_deepstream_test.txt")