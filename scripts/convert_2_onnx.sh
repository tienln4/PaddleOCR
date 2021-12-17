MODEL_DIR=/data/tienln/workspace/paddle/PaddleOCR-1/output/model
SAVE_DIR=/data/tienln/workspace/video-analytics/models
eval "$(conda shell.bash hook)"
conda activate ocr
paddle2onnx --model_dir $MODEL_DIR \
            --model_filename $MODEL_DIR/inference.pdmodel \
            --params_filename $MODEL_DIR/inference.pdiparams \
            --save_file $SAVE_DIR/paddle_r34_3x32x100_new.onnx \
            --opset_version 10 \
            --enable_onnx_checker True \
