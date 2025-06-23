#!/bin/sh

#SBATCH --requeue
#SBATCH --partition=gpu-v100
#SBATCH --gpus=1
##SBATCH --partition=idle
#SBATCH --job-name="onr"
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
##SBATCH --ntasks-per-node=1
#SBATCH --time=07-00:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --export=ALL
##SBATCH --constraint=t4
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
vpkg_require anaconda/2024.02:python3
source activate onr
source /opt/shared/slurm/templates/libexec/common.sh
vpkg_require cuda/11.6.2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export HYDRA_FULL_ERROR=1
 srun python3 -u -m src.train trainer=gpu experiment=mnist_segmentation data.dataset_type=FashionMNIST data.unify_fg_objects_intensity=False data.background_classifier=/lustre/cniel/onr/logs/eval/runs/2024-06-21_08-43-52/ae_gmm/15

