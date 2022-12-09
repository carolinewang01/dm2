#!/bin/bash
#SBATCH --job-name qm_6hv8z                                      # Job name
### Logging
#SBATCH --output=/scratch/cluster/clw4542/marl_results/qmix_6hv8z/6hv8z_%A_%a.out           # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/clw4542/marl_results/qmix_6hv8z/6hv8z_%A_%a.err            # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job
### Node info
#SBATCH --partition dgx                                                      # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 120:00:00                                                      # Run time (hh:mm:ss)
#SBATCH --gres=gpu:1                                                         # Number of gpus needed
#SBATCH --mem=120G                                                            # Memory requirements
#SBATCH --cpus-per-task=8                                                    # Number of cpus needed per task


python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=6h_vs_8z --seed=112358 &
python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=6h_vs_8z --seed=1285842 &
python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=6h_vs_8z --seed=78590 &
python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=6h_vs_8z --seed=119527 &
python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=6h_vs_8z --seed=122529 &

wait
# done
echo "Done"
exit 0