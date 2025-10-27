#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 config_file"
  exit 1
fi

# Store the first argument in a variable
input="$1"
log="log/${input}.out"

cat <<EOF > "sbatch.job"
#!/bin/bash

module load anaconda3
conda activate /ocean/projects/cis250019p/jye9/conda/envs/11685

python train.py --config configs/${input}.yaml
EOF

sbatch -o "$log" -t 30:00:00 --gpus=v100-32:1 -p GPU-shared batch.job -A cis251142