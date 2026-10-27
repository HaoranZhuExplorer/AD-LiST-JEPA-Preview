#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=raymodel
#SBATCH --chdir=<your_path>/Occ4cast_mine/data_processing/raymodel
#SBATCH --output=log/raymodel_%A_%a.out
#SBATCH --error=log/raymodel_%A_%a.err
#SBATCH --gres=gpu:v100:1
#SBATCH --account=<your_account>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>

START=0
A_PRE=70
A_POST=70
P_PRE=10
P_POST=10
INPUT="<your_path>/AD-LiST-JEPA/data/waymo/densified_points"

module purge


cd "<your_path>/Occ4cast_mine/data_processing/raymodel"

singularity exec --nv \
  --overlay "<your_path>/Occ4cast_mine/occ4cast.ext3:ro" \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "set -euo pipefail
    source /ext3/env.sh
    conda activate occ4cast

    echo \"[SLURM] task_id=\${SLURM_ARRAY_TASK_ID}\"
    echo \"[SLURM] input=${INPUT}\"

    python ray_model.py \
      -i '${INPUT}' \
      -s ${START} \
      -b \${SLURM_ARRAY_TASK_ID} \
      --a_pre ${A_PRE} --a_post ${A_POST} \
      --p_pre ${P_PRE} --p_post ${P_POST}
  "
