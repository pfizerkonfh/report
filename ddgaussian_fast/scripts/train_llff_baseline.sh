time="260530_1"
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
base="output/${time}"

# llff 3-views
BASE_PATH="${base}/llff3"
python train.py -s ./datasets/nerf_llff_data/fern -m ${BASE_PATH}/fern --eval  --n_views 3  --lowfeature_lr 0.0001 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/flower -m ${BASE_PATH}/flower --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/fortress -m ${BASE_PATH}/fortress --eval  --n_views 3 --lowfeature_lr 0.0001 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/horns -m ${BASE_PATH}/horns --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/leaves -m ${BASE_PATH}/leaves --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/orchids -m ${BASE_PATH}/orchids --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/room -m ${BASE_PATH}/room --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/trex -m ${BASE_PATH}/trex --eval  --n_views 3 --lowfeature_lr 0.0001  --grad_abs_thresh 0.003 -r 8

for scene in "${SCENES[@]}"; do
    python render.py -m "$BASE_PATH/$scene" -r 8
    python metrics.py -m "$BASE_PATH/$scene"
done

python stat_metrics.py --root_dir "$BASE_PATH" --output_file "${BASE_PATH}/metrics_summary.txt"

#python render.py -m ${BASE_PATH}/fern -r 8
#python render.py -m ${BASE_PATH}/flower -r 8
#python render.py -m ${BASE_PATH}/fortress -r 8
#python render.py -m ${BASE_PATH}/horns -r 8
#python render.py -m ${BASE_PATH}/leaves -r 8
#python render.py -m ${BASE_PATH}/orchids -r 8
#python render.py -m ${BASE_PATH}/room -r 8
#python render.py -m ${BASE_PATH}/trex -r 8
#python metrics.py -m ${BASE_PATH}/fern
#python metrics.py -m ${BASE_PATH}/flower
#python metrics.py -m ${BASE_PATH}/fortress
#python metrics.py -m ${BASE_PATH}/horns
#python metrics.py -m ${BASE_PATH}/leaves
#python metrics.py -m ${BASE_PATH}/orchids
#python metrics.py -m ${BASE_PATH}/room
#python metrics.py -m ${BASE_PATH}/trex

#python metric.py --path ${BASE_PATH}

# llff 6-views
BASE_PATH="${base}/llff6"
python train.py -s ./datasets/nerf_llff_data/fern -m ${BASE_PATH}/fern --eval  --n_views 6  --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/flower -m ${BASE_PATH}/flower --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/fortress -m ${BASE_PATH}/fortress --eval  --n_views 6 --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/horns -m ${BASE_PATH}/horns --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/leaves -m ${BASE_PATH}/leaves --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/orchids -m ${BASE_PATH}/orchids --eval  --n_views 6  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/room -m ${BASE_PATH}/room --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/trex -m ${BASE_PATH}/trex --eval  --n_views 6   --grad_abs_thresh 0.003 -r 8

for scene in "${SCENES[@]}"; do
    python render.py -m "$BASE_PATH/$scene" -r 8
    python metrics.py -m "$BASE_PATH/$scene"
done

python stat_metrics.py --root_dir "$BASE_PATH" --output_file "${BASE_PATH}/metrics_summary.txt"

#python render.py -m ${BASE_PATH}/fern -r 8
#python render.py -m ${BASE_PATH}/flower -r 8
#python render.py -m ${BASE_PATH}/fortress -r 8
#python render.py -m ${BASE_PATH}/horns -r 8
#python render.py -m ${BASE_PATH}/leaves -r 8
#python render.py -m ${BASE_PATH}/orchids -r 8
#python render.py -m ${BASE_PATH}/room -r 8
#python render.py -m ${BASE_PATH}/trex -r 8
#python metrics.py -m ${BASE_PATH}/fern
#python metrics.py -m ${BASE_PATH}/flower
#python metrics.py -m ${BASE_PATH}/fortress
#python metrics.py -m ${BASE_PATH}/horns
#python metrics.py -m ${BASE_PATH}/leaves
#python metrics.py -m ${BASE_PATH}/orchids
#python metrics.py -m ${BASE_PATH}/room
#python metrics.py -m ${BASE_PATH}/trex

#python metric.py --path ${BASE_PATH}

# llff 9-views
BASE_PATH="${base}/llff9"
python train.py -s ./datasets/nerf_llff_data/fern -m ${BASE_PATH}/fern --eval  --n_views 9 --highfeature_lr 0.005  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/flower -m ${BASE_PATH}/flower --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/fortress -m ${BASE_PATH}/fortress --eval  --n_views 9  --highfeature_lr 0.005 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/horns -m ${BASE_PATH}/horns --eval  --n_views 9 --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/leaves -m ${BASE_PATH}/leaves --eval  --n_views 9  --grad_abs_thresh 0.001 -r 8
python train.py -s ./datasets/nerf_llff_data/orchids -m ${BASE_PATH}/orchids --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8
python train.py -s ./datasets/nerf_llff_data/room -m ${BASE_PATH}/room --eval  --n_views 9  --grad_abs_thresh 0.001 -r 8
python train.py -s ./datasets/nerf_llff_data/trex -m ${BASE_PATH}/trex --eval  --n_views 9  --grad_abs_thresh 0.003 -r 8

for scene in "${SCENES[@]}"; do
    python render.py -m "$BASE_PATH/$scene" -r 8
    python metrics.py -m "$BASE_PATH/$scene"
done

python stat_metrics.py --root_dir "$BASE_PATH" --output_file "${BASE_PATH}/metrics_summary.txt"

#python render.py -m ${BASE_PATH}/fern -r 8
#python render.py -m ${BASE_PATH}/flower -r 8
#python render.py -m ${BASE_PATH}/fortress -r 8
#python render.py -m ${BASE_PATH}/horns -r 8
#python render.py -m ${BASE_PATH}/leaves -r 8
#python render.py -m ${BASE_PATH}/orchids -r 8
#python render.py -m ${BASE_PATH}/room -r 8
#python render.py -m ${BASE_PATH}/trex -r 8
#python metrics.py -m ${BASE_PATH}/fern
#python metrics.py -m ${BASE_PATH}/flower
#python metrics.py -m ${BASE_PATH}/fortress
#python metrics.py -m ${BASE_PATH}/horns
#python metrics.py -m ${BASE_PATH}/leaves
#python metrics.py -m ${BASE_PATH}/orchids
#python metrics.py -m ${BASE_PATH}/room
#python metrics.py -m ${BASE_PATH}/trex

#python metric.py --path ${BASE_PATH}
