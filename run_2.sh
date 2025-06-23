#!/bin/sh

#SBATCH --requeue
#SBATCH --partition=gpu-v100
#SBATCH --gpus=1
##SBATCH --partition=idle
#SBATCH --job-name="onr"
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --time=02-15:01:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --export=ALL
##SBATCH  --constraint=v100
#SBATCH --signal=SIGUSR1@90



cpus=""
gpus=""
lr=""

# Initialize an array to capture unknown options and their values
unknown_options=()

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --cpus-per-task)
      cpus="$2"
      shift 2
      ;;
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --lr)
      lr="$2"
      shift 2
      ;;
    *)
      # Capture unknown options and their values
      unknown_options+=("$1")
      shift
      ;;
  esac
done

#
# [EDIT] Define a Bash function and set this variable to its
#        name if you want to have the function called when the
#        job terminates (time limit reached or job preempted).
#

# Extract the variables passed as arguments

job_exit_handler() {
  if [ -r requeue-count ]; then
    C="$(cat requeue-count)"
  else
    C=0
  fi
  C=$((C + 1))
  echo $C >requeue-count
  echo "In iteration $C..."
  if [ $C -gt 20 ]; then
    echo "Job reached max requeue limit."
    exit 0
  fi
  scontrol requeue ${SLURM_JOB_ID}
  echo "Job requeued!"
  exit 0
}
export UD_JOB_EXIT_FN=job_exit_handler
export UD_JOB_EXIT_FN_SIGNALS="SIGTERM"
vpkg_require anaconda/5.3.1:python3
source activate onr
source /opt/shared/slurm/templates/libexec/common.sh
vpkg_require cuda/11.6.2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
srun python3 -u -m src.train trainer=gpu data=mnist model=perturbed_gan data.dataset_type=FashionMNIST data.unify_fg_objects_intensity=False data.num_workers=10 trainer.devices=1 data.batch_size=32 trainer.max_epochs=100 model.n_critic=1
#ckpt_path=/lustre/cniel/onr/logs/train/runs/2023-11-16_15-21-17/checkpoints/last.ckpt

#model.model_path=/lustre/cniel/onr/logs/train/runs/2023-11-15_15-45-53/ae_gmm/10

#model.model_path=/lustre/cniel/onr/logs/eval/runs/2023-11-15_17-28-30/ae_gmm/20
#2023-11-11_22-11-51 reconstruction loss for FG and BG F MNIST + divergence loss.
#model.model_path='/lustre/cniel/onr/logs/eval/runs/2023-11-11_13-55-17/ae_gmm/19'occlusion AE 19 classes.

#ckpt_path=/lustre/cniel/onr/logs/train/runs/2023-10-19_22-15-44/checkpoints/last.ckpt 
#data.dataset_type=MNIST data.random_resizing_shifting=False data.order_background_labels=False
