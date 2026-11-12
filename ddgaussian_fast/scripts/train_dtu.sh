exp_name='dtu_0527'
scenes=("scan8" "scan21" "scan30" "scan31" "scan34" "scan38" "scan40" "scan41" "scan45" "scan55" "scan63" "scan82" "scan103" "scan110" "scan114")
dataset_path='datasets/dtu'
n_views=3

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train_dtu_mask.py -s $dataset_path/$scene/ \
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
