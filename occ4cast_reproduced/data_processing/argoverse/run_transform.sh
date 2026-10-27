DATA_DIR="<your_path>/argoverse-tracking/trainval"
OUT_DIR="<your_path>/argoverse_kitti"

python transform.py --dataset_dir $DATA_DIR -o $OUT_DIR