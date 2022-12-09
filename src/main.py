import sys
import os
from os.path import dirname, abspath
import collections
from copy import deepcopy
import yaml

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th

from utils.logging import get_logger
from run import run as run_default
from run_ippo import run as run_ippo


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
results_path = os.path.join("/scratch/cluster/clw4542/marl_results", "results")

run_fcns = {
        "run_default": run_default,
        "run_ippo": run_ippo
            }

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"] # overrides env_args seed
    # run the framework
    run = run_fcns[config["run_script"]]
    print("RUN FUNCTION IS ", config["run_script"])
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def set_arg(config_dict, params, arg_name, arg_type):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            arg_name = _v.split("=")[0].replace("--", "")
            arg_value = _v.split("=")[1]
            config_dict[arg_name] = arg_type(arg_value)
            del params[_i]
            return config_dict


def recursive_dict_update(d, u):
    '''update dict d with items in u recursively. if key present in both d and u, 
    value in d takes precedence. 
    '''
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            if k not in d.keys():
                d[k] = v                
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml or attn_<>.yaml
    config_dict = _get_config(params, "--config", "")

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--alg-config", "algs")

    # update env_config and alg_config with values in config dict 
    # copy modified env args for logging purpose 
    config_dict = recursive_dict_update(config_dict, {**env_config, **alg_config})

    # overwrite seed 
    config_dict = set_arg(config_dict, params, "--seed", int)

    # add config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

