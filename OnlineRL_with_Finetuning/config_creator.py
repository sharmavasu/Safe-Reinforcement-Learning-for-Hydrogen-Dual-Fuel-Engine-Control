"""Script for creating the universal DDPG master's configuration file.

File:   config_creator.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-08-24
"""

import sys
import argparse
import copy
import json

from typing import Any


# Parse command line arguments
arg_parser = argparse.ArgumentParser(
    description=(
        "Export the Universal DDPG"
        " LExCI 2 Master's configuration to a JSON file."
    )
)
arg_parser.add_argument("output_file", type=str, help="Output file to write.")
cli_args = arg_parser.parse_args(sys.argv[1:])


# Master configuration dictrionary
master_config = {}
# =========================== MAKE ADJUSTMENTS HERE ============================#
master_config["obs_size"] = 16
master_config["action_size"] = 4
master_config["addr"] = "127.0.0.1"
master_config["port"] = 5555
master_config["num_experiences_per_cycle"] = 10
master_config["mailbox_buffer_size"] = 1 * 1024**3
master_config["min_num_minions"] = 1
master_config["max_num_minions"] = 1
master_config["minion_job_timeout"] = 3600.0
master_config["minion_params"] = {
    "INITIAL_STDEV": 0.0597, #0.2 #0.1392 # 0.0911
    "STDEV_DECAY_FACTOR": 0.985, #0.975
    "TENSOR_ARENA_SIZE": 100000,
}
master_config["nn_format"] = "tflite"
master_config["nn_size"] = 64 * 1024**1
master_config["output_dir"] = "~/lexci_results/onlineRL/TD3"
master_config["b_save_training_data"] = True
master_config["b_save_sample_batches"] = True
master_config["validation_interval"] = 4
master_config["num_replay_trainings"] = 2
master_config["perc_replay_trainings"] = 0
master_config["num_exp_before_replay_training"] = (
    1 * master_config["num_experiences_per_cycle"]
)
master_config["offline_data_import_folder"] = ""#/home/vasu3/lexci_results/onlineRL/TD3/DataSamples_12_11_14_34_and_12_13_13_23"#/home/vasu3/lexci_results/onlineRL/TD3/2024-12-11_14-34-24.475299/Sample_Batch_JSONs/"#/home/vasu3/lexci_results/onlineRL/TD3/2024-11-25_15-23-27.903310/Sample_Batch_JSONs/"#/home/vasu3/lexci_results/OfflineData/"  # "/home/pi/lexci_results/OfflineData_VSR30090"
master_config["b_offline_training_only"] = False
master_config["checkpoint_file"] = ""#/home/vasu3/lexci_results/pretraining/2024-10-18_10-29-49.673115/Checkpoints/checkpoint-50"  
master_config["model_h5_folder"] = "/home/vasu3/lexci_results/onlineRL/TD3/2024-12-20_13-35-03.136264/NN_h5/Cycle_28/"#/home/vasu3/lexci_results/onlineRL/TD3/2024-12-13_13-23-59.381195/NN_h5/Cycle_28"#"/home/vasu3/lexci_results/onlineRL/TD3/2024-12-11_14-34-24.475299/NN_h5/Cycle_24"#/home/vasu3/lexci_results/pretraining/2024-11-25_09-44-42.443015/NN_h5/Cycle_50/"#/home/vasu3/lexci_results/pretraining/2024-11-25_09-44-42.443015/NN_h5/Cycle_50/"#/home/vasu3/lexci_results/pretraining/2024-11-14_17-27-09.263057/NN_h5/Cycle_40/"#/home/vasu3/lexci_results/pretraining/2024-10-18_10-29-49.673115/NN_h5/Cycle_50/"#"/home/vasu3/lexci_results/onlineRL/2024-11-07_12-10-34.804511/NN_h5/Cycle_10/"
# If the documentation string isn't empty, the universal Master will create a
# text file called 'Documentation.txt' in the training's log directory and write
# the content of the string into said file.
master_config["doc"] = ""
# ==============================================================================#


# PPO configuration dictionary
import ray.rllib.agents.ddpg as ddpg

ddpg_config = copy.deepcopy(ddpg.DEFAULT_CONFIG)
# =========================== MAKE ADJUSTMENTS HERE ============================#
ddpg_config["actor_hiddens"] = [64,64]
ddpg_config["actor_hidden_activation"] = "relu"
ddpg_config["critic_hiddens"] = [64,64]
ddpg_config["critic_hidden_activation"] = "relu"
ddpg_config["replay_buffer_config"][
    "capacity"
] = 100000  # 65000 (Offline RL) 50000 (Online RL)
ddpg_config["store_buffer_in_checkpoints"] = True
ddpg_config["train_batch_size"] = 300  # 1024 (Offline RL) 64 (Online RL)
ddpg_config["gamma"] = 0.99  # 0.1(Offline RL) 0.9 (Online RL)
ddpg_config["actor_lr"] = 1e-3  # 1e-5 (Offline RL) 1e-3 (Online RL)
ddpg_config["critic_lr"] = 1e-3 # 1e-5 (Offline RL) 1e-3 (Online RL)
# Update target networks using `tau*policy + (1 - tau)*target_policy`
ddpg_config["tau"] = 0.001  # 0.0001 (Offline RL) 0.001 (Online RL)
ddpg_config["target_network_update_freq"] = 2
ddpg_config["grad_clip"] = 2
ddpg_config["l2_reg"] = 1e-5
ddpg_config["twin_q"] = True
ddpg_config["policy_delay"] = 2
ddpg_config["smooth_target_policy"] = True
ddpg_config["target_noise_clip"] = 0.3
ddpg_config["target_noise"] = 0.1
# ==============================================================================#
# Remove keys that aren't JSON-serializable
keys_to_remove = []
for k, v in ddpg_config.items():
    if v is not None and type(v) not in [dict, list, str, int, float, bool]:
        print(
            f"Removing key '{k}' with value '{v}' from the DDPG configuration"
            + " as it isn't JSON-serializable."
        )
        keys_to_remove.append(k)
for k in keys_to_remove:
    del ddpg_config[k]


# Write the JSON file
config = {"master_config": master_config, "ddpg_config": ddpg_config}
with open(cli_args.output_file, "w") as f:
    json.dump(config, f, indent=2)
    
 #/home/vasu3/lexci_results/pretraining/2024-11-21_10-38-48.164098/NN_h5/Cycle_40/: 128,64 networks: works well
 #/home/vasu3/lexci_results/pretraining/2024-11-08_10-19-59.289769/NN_h5/Cycle_60/ : Excellent first shot
# /home/vasu3/lexci_results/pretraining/2024-11-14_17-27-09.263057/NN_h5/Cycle_40/ : Best for learning
# /home/vasu3/lexci_results/pretraining/2024-11-25_09-44-42.443015/NN_h5/Cycle_50/ : Good for learning
#/home/vasu3/lexci_results/onlineRL/TD3/2024-11-25_15-23-27.903310/NN_h5/Cycle_52/ : Just online Data
# /home/vasu3/lexci_results/onlineRL/TD3/2024-12-20_13-35-03.136264/NN_h5/Cycle_28/ : 

