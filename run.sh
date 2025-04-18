#!/bin/bash
#SBATCH --partition=rtx8000
#SBATCH --job-name=NLP_ls
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH -o /home/l.cai/job_logs/train_%A_%a.out
#SBATCH -e /home/l.cai/job_logs/train_%A_%a.err




nvidia-smi

echo "starting training"
rsync -avu /data/groups/beets-tan/l.cai/rectal_nlp/breast_reports /processing/l.cai/

python /data/groups/beets-tan/l.cai/rectal_nlp/re_train.py