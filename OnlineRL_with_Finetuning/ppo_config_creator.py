"""Script for creating the universal PPO master's configuration file.

File:   lexci2/universal_masters/universal_ppo_master/ppo_config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-12


Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import sys
import argparse
import copy
import json

from typing import Any


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Export the Universal PPO"
        " LExCI 2 Master's configuration to a JSON file."
    )
)
arg_parser.add_argument("output_file", type=str, help="Output file to write.")
cli_args = arg_parser.parse_args(sys.argv[1:])


# Master configuration dictrionary
master_config = {}
# =========================== MAKE ADJUSTMENTS HERE ===========================#
master_config["obs_size"] = 16
master_config["action_size"] = 4
master_config["action_type"] = "continuous"
master_config["addr"] = "0.0.0.0"
master_config["port"] = 5555
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 2
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results/onlineRL/PPO"
master_config["b_save_training_data"] = True
master_config["b_save_sample_batches"] = False
master_config["validation_interval"] = 10
master_config["checkpoint_file"] = "/home/vasu3/lexci_results/pretraining/PPO/2024-11-22_11-46-02.112594/Checkpoints/checkpoint-60"
master_config["model_h5_folder"] = "" #/home/vasu3/lexci_results/pretraining/PPO/2024-11-22_11-46-02.112594/NN_h5/Cycle_60/"
# If the documentation string isn't empty, the universal Master will create a
# text file called 'Documentation.txt' in the training's log directory and write
# the content of the string into said file.
master_config["doc"] = ""
# =============================================================================#


# PPO configuration dictionary
import ray.rllib.agents.ppo as ppo

ppo_config = copy.deepcopy(ppo.DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ===========================#
ppo_config["model"]["fcnet_hiddens"] = [16, 16, 16]
ppo_config["model"]["fcnet_activation"] = "tanh"
ppo_config["train_batch_size"] = 600  # = Number of experiences per cycle
ppo_config["sgd_minibatch_size"] = 512
ppo_config["num_sgd_iter"] = 8
ppo_config["gamma"] = 0.99
ppo_config["clip_param"] = 0.3
ppo_config["vf_clip_param"] = 1e6
ppo_config["lr"] = 0.001 #0.0025 
ppo_config["grad_clip"] = 1
ppo_config["entropy_coeff"] = 0.001
# =============================================================================#
# Remove keys that aren't JSON-serializable
keys_to_remove = []
for k, v in ppo_config.items():
    if v is not None and type(v) not in [dict, list, str, int, float, bool]:
        print(
            f"Removing key '{k}' with value '{v}' from the PPO configuration as"
            + " it isn't JSON-serializable."
        )
        keys_to_remove.append(k)
for k in keys_to_remove:
    del ppo_config[k]


# Write the JSON file
config = {"master_config": master_config, "ppo_config": ppo_config}
with open(cli_args.output_file, "w") as f:
    json.dump(config, f, indent=2)
