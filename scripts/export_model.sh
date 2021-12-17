MODEL_DIR=output/plate_rec_3x32x100

python3.8 tools/export_model.py \
    -c ./$MODEL_DIR/config.yml \
    -o Global.pretrained_model=./$MODEL_DIR/latest  \
    Global.save_inference_dir=./output/model/