# python3.8 tools/infer/predict_system.py \
#     --image_dir="./images/img/" \
#     --det_model_dir="./model/server/det/" \
#     --rec_model_dir="./model/server/rec/" \
#     --use_angle_cls=false \
#     --use_gpu=true

python3.8 tools/infer/predict_rec.py \
        --image_dir="test_.jpg" \
        --rec_model_dir="./model/plate/rec_khanhtq" \
        --use_gpu=false \
        --warmup=false

# python3.8 tools/infer/predict_det.py \
#         --image_dir="./20201213_081633"  \
#         --det_model_dir="./model/server/det/"\
#         --use_gpu=true





