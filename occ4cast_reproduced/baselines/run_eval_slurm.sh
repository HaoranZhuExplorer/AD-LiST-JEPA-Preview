#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:20:00
#SBATCH --mem=96GB
#SBATCH --job-name=4docc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --output=log/eval_%j.out
#SBATCH --error=log/eval_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --account=<your_account>

cd /scratch/$USER/Occ4cast_mine/baselines

singularity exec --nv \
	    --overlay "<your_path>/Occ4cast_mine/occ4cast.ext3:ro" \
    	--overlay "<your_path>/Occ4cast_mine/stored_data/OCFBench-Lyft/Lyft-full.sqf:ro" \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; \
					 conda activate occ4cast; \
					 python eval.py -r /Occ4D-Lyft \
					 -d '<your_path>/Occ4cast_mine/baselines/results/conv3d_lyft_p55_lr0.0005_batch6'"