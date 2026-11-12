exp_name='mip12'
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
dataset_path='datasets/mipnerf360'
n_views=12

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m output/$exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views

  echo "Rendering $scene..."
  python render.py -m output/$exp_name/$scene -r 8
  python metrics.py -m output/$exp_name/$scene
done

# Compute metrics for all scenes
python metric.py --path output/$exp_name

python stat_metrics.py --root_dir "output/$exp_name" --output_file "output/$exp_name/metrics_summary.txt"
