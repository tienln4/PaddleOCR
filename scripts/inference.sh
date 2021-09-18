# python3.8 tools/infer/predict_system.py \
#     --image_dir="./images/img/" \
#     --det_model_dir="./model/server/det/" \
#     --rec_model_dir="./model/server/rec/" \
#     --use_angle_cls=false \
#     --use_gpu=true

python3.8 tools/infer/predict_rec.py \
        --image_dir="benchmark/X/images" \
        --rec_model_dir="./model/plate/rec/" \
        --use_gpu=true \
        --warmup=false

# python3.8 tools/infer/predict_det.py \
#         --image_dir="./images/img/14c0520858ad41139638f5766b796e0e.jpg"  \
#         --det_model_dir="./model/server/det/"\
#         --use_gpu=true





