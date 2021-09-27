from .config import Config


class PaddleRecognizerConfig(Config):
    @property
    def use_gpu(self):
        return self.config["global"]["use_gpu"]

    @property
    def ir_optim(self):
        return self.config["global"]["ir_optim"]

    @property
    def use_tensorrt(self):
        return self.config["global"]["use_tensorrt"]

    @property
    def use_fp16(self):
        return self.config["global"]["use_fp16"]

    @property
    def gpu_mem(self):
        return self.config["global"]["gpu_mem"]

    @property
    def rec_algorithm(self):
        return self.config["recognizer"]["rec_algorithm"]

    @property
    def rec_model_dir(self):
        return self.config["recognizer"]["rec_model_dir"]

    @property
    def rec_image_shape(self):
        return self.config["recognizer"]["rec_image_shape"]

    @property
    def rec_char_type(self):
        return self.config["recognizer"]["rec_char_type"]

    @property
    def rec_batch_num(self):
        return self.config["recognizer"]["rec_batch_num"]

    @property
    def max_text_length(self):
        return self.config["recognizer"]["max_text_length"]

    @property
    def rec_char_dict_path(self):
        return self.config["recognizer"]["rec_char_dict_path"]

    @property
    def use_space_char(self):
        return self.config["recognizer"]["use_space_char"]

    @property
    def vis_font_path(self):
        return self.config["recognizer"]["vis_font_path"]

    @property
    def drop_score(self):
        return self.config["recognizer"]["drop_score"]

    @property
    def use_zero_copy_run(self):
        value = self.config["recognizer"]["use_zero_copy_run"]
        return value

    @property
    def enable_mkldnn(self):
        value = self.config["global"]["enable_mkldnn"]
        return value
    
    @property
    def dataset(self):
        value = self.config["benchmark"]["dataset"]
        return value
    
    @property
    def benchmark(self):
        value = self.config["benchmark"]["benchmark"]
        return value
    
    @property
    def result(self):
        value = self.config["benchmark"]["result"]
        return value

    @property
    def max_batch_size(self):
        value = self.config["global"]["max_batch_size"]
        return value

    @property
    def min_subgraph_size(self):
        value = self.config["global"]["min_subgraph_size"]
        return value

