This repository was originally forked from https://github.com/oxwhirl/pymarl. 
Please refer to their READMe.md for instructions on how to install StarCraft II and SMAC, save/load checkpointed models, visualize SC2 tasks.
Parts of the IPPO/DM2 implementation originate from https://github.com/marlbenchmark/on-policy. 

Below are the instructions to reproduce the experiments found in the paper, [**DM^2**: Distributed Multi-Agent Reinforcement Learning for Distribution Matching](https://arxiv.org/abs/2206.00233). The [RMAPPO](https://arxiv.org/abs/2103.01955) baseline was produced by directly running the code provided by Yu et al. at https://github.com/marlbenchmark/on-policy.
Please contact us if you have any further questions. 

All commands below should be run from the `pymarl` directory. The list of seeds used in the main paper are as follows: 112358, 1285842, 78590, 119527, 122529.
The maps used in the paper are `5v6`, `3sv4z`, and `3sv3z`. The commands below will use the `5v6` map as an example, but different map names may be substituted in.
Different algorithms may be run by modifying the correct `.yaml` file in `pymarl/src/config/default<>.yaml`, as specified below.

# Running IPPO
In the file `default_ippo_5v6.yaml`, make the following modifications:

- Set `rew_type: "env"`
- Set `update_gail: False`

All results will be stored in the `local_results_path`.
To save IPPO demonstrations, see the below subsection .

Run the following command:
```shell
python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=<seed>
````

## Sampling Demonstrations
To sample IPPO demonstrations, set the following additional parameters in the same `default_ippo_5v6.yaml`:
- Set `save_agent_batches: False`
- Set `save_agent_batchsize` to the number of desired demonstration state-action pairs. The results in the paper were computed with state-only demonstrations, and each algorithm (where relevant) had access to 1000 demonstration states, unless specified otherwise.
- Set `save_agent_batches_interval` to the desired saving interval. By default, this value is 1M timesteps.

# Running QMIX
No modifications need to be made to `default.yaml`.
All results will be stored in the `local_results_path`. 
To save QMIX demonstrations, see the below subsection .

Run the following command:

```shell
python src/main.py --env-config=sc2 --config=default --alg-config=qmix with env_args.map_name=5m_vs_6m --seed=<seed>
````

## Sampling Demonstrations
The QMIX demonstrations used the paper experiments were sampled from fully trained QMIX policies. 
To sample QMIX demonstrations, first train a QMIX policy on the desired map. Next, make the following modifications to `default_qmix_savetraj.yaml`:
- Set `checkpoint_path`  to the path of the saved QMIX policy checkpoints.
- Set `epsilon_start` to the same value as `epsilon_finish` (refer to the Appendix of the paper for epsilon values). In practice, for DM^2 to learn well from QMIX demonstrations, it is sometimes important for there to be a degree of random action noise in the demonstrations. 

Run the following command:
```shell
python src/main.py --env-config=sc2 --config=default_qmix_savetraj --alg-config=qmix with env_args.map_name=5m_vs_6m --seed=<seed>
````

# Running DM2
First, follow the above instructions to sample demonstrations. These demonstrations may be sampled from QMIX or IPPO checkpoints.

Next, make the following modifications to `default_ippo_5v6.yaml`:
- Set `rew_type: "mixed"`  
- Set `update_gail` to True
- Set `gail_data_paths` to a list that contains the path to the demonstration data directory. For DM^2, the list should contain only a single path.
- Set `gail_exp_eps` to the number of demonstration states desired. The results presented in the paper used 1000. 
- Set `gail_exp_use_same_data_idxs: True`
- Set `gail_exp_use_single_seed: True`

All results will be stored in the `local_results_path`; update this if necessary.

Finally, run the following command:
```shell
python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=<seed>
````

# Running DM2 with Self Imitation Learning 
No demonstrations are required. Make the following modifications to `default_ippo_5v6.yaml`:
- Set `rew_type: "mixed"`  
- Set `update_gail: True` 
- Set `gail_buffer_size` to the number of episodes that you would like to store in the SIL buffer.
- Set `gail_sil` to True

# Running Ablations of DM2
Running the ablations requires access to demonstration data as well. To run all 3 ablations, we require demonstration data sampled from `n_allied_agents` distinct checkpoints, and `gail_exp_eps` * `n_allied_agents` demonstration states per checkpoint. 

Modify `default_ippo_5v6.yaml` as follows: 
- Set `rew_type: "mixed"`  
- Set `update_gail` to True
- Set `gail_data_paths` to a list that contains all paths to all demonstration data; the length of this list should be equal to `n_allied_agents`.
- For the concurrently sampled ablations, set `gail_exp_use_same_data_idxs` to True. 
- For the co-trained experts ablations, set `gail_exp_use_single_seed` to True.

Run the following command:
```shell
python src/main.py --env-config=sc2 --config=default_ippo_5v6 --alg-config=ippo with env_args.map_name=5m_vs_6m --seed=<seed>
````

# Misc

Figures in the paper were generated by the notebook at `pymarl/notebooks/Paper Figures.ipynb`.
