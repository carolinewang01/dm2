import sys
import re
import os
import datetime
import yaml
import glob


map_namedict = {"5v6": "5m_vs_6m",
                "3sv4z": "3s_vs_4z",
                }

alg = 'ippo'
policy_map_names = ["5v6", "3sv4z"]
eval_seeds = [1111111]
load_steps = [
    1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000
]

checkpoint_dict = {
    "5v6": [
        "ippo_sc2_saved-batchsize=5005_seed=112358_03-06-21-12-41"
    ],
    "3sv4z": [
        "ippo_sc2_baseline_seed=119527_01-23-20-41-57"
    ]
}
results_dir = "/scratch/cluster/clw4542/marl_results/"
conf_name = 'default_ippo_5v6'
source_conf_path = f'src/config/{conf_name}.yaml'


def find_model_path(alg: str, map_name: str):
    '''Use model file with most recent time
    '''
    ckpt_model_dir = f"{results_dir}/{alg}_{map_name}/models"

    most_recent_date = None
    ckpt_model_path = None

    for f in os.listdir(ckpt_model_dir):
        if not len(f.split("__")) == 3:
            continue
        _, experiment_name, timestr = f.split("__")
        exp_name_list = experiment_name.split("_")
        if "baseline" not in exp_name_list or "train" not in exp_name_list:
            continue

        date = datetime.datetime.strptime(timestr, "%Y-%m-%d_%H-%M-%S")
        if most_recent_date is None:
            most_recent_date = date
            ckpt_model_path = f"{ckpt_model_dir}/{f}"
        elif date > most_recent_date:
            most_recent_date = date
            ckpt_model_path = f"{ckpt_model_dir}/{f}"

    return ckpt_model_path


def modify_default_yaml(policy_map_name: str,
                        alg: str, dest_conf_path: str,
                        checkpoint_model_path: str, checkpoint_model_name: str,
                        eval_seed, load_step=0):
    '''Change values in default yaml file
    '''
    with open(source_conf_path) as f:
        conf = yaml.load(f)

    conf['use_tensorboard'] = True
    conf['save_model'] = False

    conf['checkpoint_paths'] = [checkpoint_model_path]
    conf['local_results_path'] = f"{results_dir}/{alg}_{policy_map_name}_eval"
    conf['evaluate'] = True
    conf['save_replay'] = False

    conf['test_nepisode'] = 128
    conf['runner'] = 'ippo'
    conf['mac'] = 'dcntrl_mac'
    conf['load_step'] = load_step
    conf['action_selector'] = 'epsilon_greedy'

    conf['label'] = f'eval_ckpt={checkpoint_model_name.strip("/ ")}_step={str(load_step)}'

    with open(dest_conf_path, "w") as f:
        yaml.dump(conf, f)


def str_search_param(name:str, param_name:str, param_type=None):
    if param_type is float:
        param_match = re.search(f"{param_name}=\\d{1,3}(\\.\\d{1,3})?", name)
    elif param_type is int:
        param_match = re.search(f"{param_name}=\\d*", name)        
    elif param_type is str: 
        # print("SEARCHING STRING MATCH")
        param_match = re.search(f"{param_name}=[a-z=\\<\\>]*", name)

    if param_match is not None: 
        param_match = param_type(param_match.group().replace(f"{param_name}=", ""))
    return param_match


def execute_eval(alg, policy_map_name,
                          eval_seed,
                          checkpoint_model_name, checkpoint_model_path,
                          load_step):

    # create temporary yaml
    dest_conf_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    dest_conf_path = f'src/config/{dest_conf_name}.yaml'

    modify_default_yaml(policy_map_name=policy_map_name,
                        alg=alg,
                        dest_conf_path=dest_conf_path,
                        checkpoint_model_path=checkpoint_model_path,
                        checkpoint_model_name=checkpoint_name,
                        eval_seed=eval_seed, load_step=load_step)

    exec_str = f"python src/main.py --env-config=sc2 --config={dest_conf_name} --alg-config={alg} with env_args.map_name={map_namedict[policy_map_name]} --seed={eval_seed}"
    os.system(exec_str)

    # delete temporary yaml
    os.remove(dest_conf_path)


if __name__ == '__main__':
    for policy_map_name, checkpoint_name_list in checkpoint_dict.items():
        for checkpoint_name in checkpoint_name_list:
            for load_step in load_steps:
                for eval_seed in eval_seeds:
                    checkpoint_model_path = os.path.join(
                        results_dir, f"{alg}_{policy_map_name}", "models", checkpoint_name)
                    assert os.path.exists(checkpoint_model_path), f"Unable to find checkpointed model for {checkpoint_name}"                
                    execute_eval(alg=alg, policy_map_name=policy_map_name, 
                                 eval_seed=eval_seed,
                                 checkpoint_model_name=checkpoint_name, 
                                 checkpoint_model_path=checkpoint_model_path,
                                 load_step=load_step)
