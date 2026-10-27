#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=4docc_waymo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=log/4docc_%j.out
#SBATCH --error=log/4docc_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --account=<your_account>

cd /scratch/$USER/AD-LiST-JEPA/occ4cast_reproduced/baselines

singularity exec --nv \
  --overlay "<your_path>/AD-LiST-JEPA/occ4cast_reproduced/occ4cast.ext3:ro" \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "set -euo pipefail
    source /ext3/env.sh
    conda activate occ4cast

    # 1) Point everything to certifi's CA bundle (from this conda env)
    PEM=\$(python -c 'import certifi; print(certifi.where())')
    export SSL_CERT_FILE=\"\$PEM\"
    export REQUESTS_CA_BUNDLE=\"\$PEM\"
    export CURL_CA_BUNDLE=\"\$PEM\"
    export SSL_CERT_DIR=/etc/ssl/certs  # harmless if missing

    # 2) Put wandb files on scratch (recommended)
    mkdir -p /scratch/$USER/wandb/{runs,cache}
    export WANDB_DIR=/scratch/$USER/wandb/runs
    export WANDB_CACHE_DIR=/scratch/$USER/wandb/cache

    # 3) Make sure SSL is NOT disabled from earlier attempts
    unset WANDB_VERIFY_SSL

    export WANDB_MODE=online

    # ---- Launch training ----
    python train.py -d waymo \
      -m conv3d \
      -r '<your_path>/AD-LiST-JEPA/data/waymo/Occ4D-Waymo' \
      --p_pre 5 \
      --p_post 5 \
      --batch-size 1 \
      --num-workers 16 \
      --num-epoch 15 \
      --lr-start 5e-4
  "
