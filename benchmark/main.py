from benchmark.modules.configs.paddle_recognizer_config import PaddleRecognizerConfig
from benchmark.modules.recognizer.paddle_recognizer import PaddleRecognizer
from shutil import move
import numpy as np
import fastwer
import cv2
import os

class Benchmark:
    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset
        self.recognizer = PaddleRecognizer(self.config)

    def CER(self, gt, pred):
        return fastwer.score_sent(pred, gt, char_level=True)
    
    def WER(self, gt, pred):
        return fastwer.score_sent(pred, gt)

    def check(self, result):
        if result == "True":
            return 1
        else:
            return 0

    def GenPythonRecord(self):
        lb_fp = open(os.path.join(self.dataset, "gt.txt"))
        record_path = os.path.join(self.config.result, self.dataset.split("/")[-1] + "_python.txt")
        if os.path.exists(record_path):
            os.remove(record_path)

        result = open(record_path,"a")

        for line in lb_fp:
            line = line.split("\t")
            img_fp = os.path.join(self.dataset, line[0])
            img_gt = line[1][:-1]

            img = cv2.imread(img_fp)
            rec_objs = self.recognizer.recognize([img])
            
            for obj in rec_objs:
                result_log = f"{img_fp} {img_gt} {obj.text} {obj.score} \n"
                result.write(result_log)

    def GenCppRecord(self, pred_result_fp, framework,):
        if framework != "deepstream" and framework != "tensorrt":
            print ("Only support deepstream or tensorrt")
            return
        else:
            lb_fp = open(os.path.join(self.dataset, "gt.txt"))
            record_path = os.path.join(self.config.result, self.dataset.split("/")[-1] + f"_{framework}_test.txt")
            if os.path.exists(record_path):
                os.remove(record_path)

            result = open(record_path,"a")
            if os.path.exists(pred_result_fp):
                cpp_pred = open(pred_result_fp)
            else:
                print("pred_result_fp is not exists")
                return

            cpp_results = []
            for line in cpp_pred:
                line.split(" ") # Todo add score
                cpp_results.append([line[:-1], 0.9]) 
            
            for i, line in enumerate(lb_fp):
                line = line.split("\t")
                img_fp = os.path.join(self.dataset, line[0])
                img_gt = line[1][:-1]
                result_line = f"{img_fp} {img_gt} {cpp_results[i][0]} {cpp_results[i][1]}\n"
                result.write(result_line)

    def CalculateAccuracy(self, record_path):
        result_file = open(record_path)
        result = []
        for i,line in enumerate(result_file):
            line = line.split(" ")
            gt = str(line[1])
            pred = str(line[2])
            result.append([self.CER(gt, pred), self.WER(gt, pred)])

        result = np.array(result)
        print("CER: ", round(sum(result[:,0])/len(result[:,0]), 2), " %")
        print("WER: ", round(sum(result[:,1])/len(result[:,1]), 2), " %")
    
    

