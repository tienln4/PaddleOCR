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

    def MergeBenchmarkResults(self, python_record, deepstream_record, tensorrt_record):
        merge_result_file = open(os.path.join(self.config.result, "merged_result.txt"), "a")
        merge_result_view_file = open(os.path.join(self.config.result, "merged_result_view.txt"), "a")
        python_file = open(python_record)
        deepstream_file = open(deepstream_record)
        tensorrt_file = open(tensorrt_record)

        deepstream_pred_list = []
        tensorrt_pred_list = []

        for line in deepstream_file:
            deepstream_pred_list.append(line[:-1])
        
        for line in tensorrt_file:
            tensorrt_pred_list.append(line[:-1])

        
        for i, line in enumerate(python_file):
            line = line.split(" ")
            if deepstream_pred_list[i]==line[1]: ds_result = "True" 
            else: ds_result = "False"
            if tensorrt_pred_list[i]==line[1]: trt_result = "True" 
            else: trt_result = "False"

            new_line = f"{line[0]} {line[1]} {line[2]} {line[3]} {line[4]} {line[5]} {deepstream_pred_list[i]} {ds_result} {tensorrt_pred_list[i]} {trt_result} \n"
            view_line =  f'{line[0]:40s} | {line[1]:10s} | {line[2]:10} | {line[3]:5s} | {line[4]:20s} | {line[5]:20s} | {deepstream_pred_list[i]:10s} | {ds_result:5s} | {tensorrt_pred_list[i]:10s} | {trt_result:5s}|\n'
            merge_result_view_file.write(view_line)
            merge_result_file.write(new_line)

    def ExportResult(self, record_path):
        result_file = open(record_path)
        result = []

        
        for i,line in enumerate(result_file):
            if i > 0:
                line = line.split(" ")
                result.append([self.check(line[3]), self.check(line[7]), self.check(line[9])])
        result = np.array(result)
        print("Paddle: ", sum(result[:,0])/len(result[:,0]))
        print("Deepstream: ", sum(result[:,1])/len(result[:,0]))
        print("Tensorrt: ", sum(result[:,2])/len(result[:,0]))

        #         time = time + float(line[5])
        #         if line[3] == "True": true_plate += 1
        #         else : false_plate += 1
        # print(true_plate/(true_plate+false_plate))
        # print(time/347)

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
    
    

