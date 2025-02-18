"""Minion for the pendulum environment in Simulink.

File:   simulink_pendulum_minion.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2023-09-21


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


import lexci2
from lexci2.minion.minion import Minion
from lexci2.minion.controllers.matlab_simulink_controller import (
    MatlabSimulinkController,
)
from lexci2.data_containers import Experience, Episode, Cycle
from lexci2.utils.transform import transform_linear, transform_tanh

import time
import logging
import numpy as np
from typing import Any


# The algorithm that is used for training. This must be either 'ppo' or 'ddpg'.
# Also, remember to set the corresponding value in
# 'ReinforcementLearningBlockInit.m'. When using PPO, the S-Function
# 'PolicyNeuralNetwork' must have a two-dimensional output. For DDPG, its output
# is one-dimensional.
algorithm = "ddpg"


# Logger
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


# MATLAB/Simulink controller
controller = MatlabSimulinkController()


def denormalize_obs(norm_obs: np.ndarray) -> np.ndarray:
    """Transform a normalized observation into a regular observation.

    Arguments:
        - norm_obs: np.ndarray
              The normalized observation to transform.

    Returns:
        - _: np.ndarray
              The regular observation.
    """

    return transform_linear(
        norm_obs,
        np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
        np.array([+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1], dtype=np.float32),
        np.array([-20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,-20, -20, -20, -20], dtype=np.float32),
        np.array([+20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20], dtype=np.float32),
    )


def denormalize_action(norm_action: np.ndarray) -> np.ndarray:
    """Transform a normalized action into a regular action.

    Arguments:
        - norm_action: np.ndarray
              The normalized action to transform.

    Returns:
        - _: np.ndarray
              The regular action.
    """

    return transform_tanh(
        norm_action,
        np.array([-2.3201,-2.1257,-1.8309, -1.6562], dtype=np.float32),
        np.array([+1.4113, 0.9867,0.9542, 2.1874], dtype=np.float32),
    )


def calc_stdev(
    initial_stdev: float, stdev_decay_factor: float, cycle_no: int
) -> float:
    """Calculate the standard deviation of the action distribution as a function
    of the current LExCI cycle.

    Arguments:
        - initial_stdev: float
              Initial standard deviation, i.e. at cycle 0.
        - stdev_decay_factor: float
              Factor in (0, 1) that governs how quickly the standard deviation
              converges to zero.
        - cycle_no: int
              Current LExCI cycle number.

    Returns:
        - _: float
              The standard deviation of the action distribution.
    """

    return initial_stdev * stdev_decay_factor**cycle_no


def retrieve_episode(
    controller: MatlabSimulinkController, agent_id: str
) -> Episode:
    """Get the experiences that were generated during an episode.

    The Simulink model saves its experiences to the MATLAB workspace using the
    variable 'experience_buffer'.

    Arguments:
        - controller: MatlabSimulinkController
              A controller for the Simulink model of the environment.
        - agent_id: str
              The ID of the agent that was used when generating the data.

    Returns:
        - _: Episode
              An episode containing the collected experiences.
    """

    # Transfer the data from MATLAB to Python
    controller.run_cmd("experience_buffer.get('norm_obs');")
    norm_obs = controller.read_workspace_var("ans")
    # print('OBSERVATION')
    # print(len(norm_obs)) #626
    # print(norm_obs[500])
    controller.run_cmd("experience_buffer.get('norm_action');")
    norm_action = controller.read_workspace_var("ans")
    # print('ACTION')
    # print(len(norm_action)) #4
    # print(norm_action[3][0])    
    # print(norm_action[3][0][1])    

    #print(len(norm_action[0][0])) #126
    #print((norm_action[0][0]))
    #print((norm_action[0]))

    controller.run_cmd("experience_buffer.get('new_norm_obs');")
    new_norm_obs = controller.read_workspace_var("ans")
    # print('OBSERVATION_NORM')
    # print(len(new_norm_obs))#626
    # print(new_norm_obs[500])


    controller.run_cmd("experience_buffer.get('reward');")
    reward = controller.read_workspace_var("ans")
    # print('Reward')
    #print((reward[2]))
    #print((reward[2][0]))
    # print(len(reward))
    # print((reward))
    # print((reward[2]))

    controller.run_cmd("experience_buffer.get('done');")
    done = controller.read_workspace_var("ans")
    # print('DONE')
    # print(len(done))
    # print(done[2])

    controller.run_cmd("experience_buffer.get('norm_action_dist');")
    norm_action_dist = controller.read_workspace_var("ans")
    # print('NORM_ACTION_DIST')
    # print(len(norm_action_dist))#4 
    # print(len(norm_action_dist[0][0])) #626
    # print((norm_action_dist[0][0]))

    controller.run_cmd("experience_buffer.get('IMEP_RL');")
    IMEP_RL = controller.read_workspace_var("ans")
    # print(len(IMEP_RL))

    controller.run_cmd("experience_buffer.get('IMEP');")
    IMEP = controller.read_workspace_var("ans")
    # print(len(IMEP))
    # print(IMEP)
    # print(IMEP[0])
    # print(IMEP[0][0])
    # print(IMEP[0][1])

    controller.run_cmd("experience_buffer.get('Rand_0');")
    Rand_0 = controller.read_workspace_var("ans")   
    # print('RANDOMS')
    # print(Rand_0)

    controller.run_cmd("experience_buffer.get('Rand_1');")
    Rand_1 = controller.read_workspace_var("ans")
    # print(Rand_1)

    controller.run_cmd("experience_buffer.get('Rand_2');")
    Rand_2 = controller.read_workspace_var("ans")
    # print(Rand_2)

    controller.run_cmd("experience_buffer.get('Rand_3');")
    Rand_3 = controller.read_workspace_var("ans")
    # print(Rand_3)

    experiences = []
    # The very first experience is meant to be skipped as it contains faulty
    # initial values
    '''
    for i in range(1, len(norm_obs[0][0])):
        if algorithm == "ddpg":
            aux_data = {
                "norm_action_dist[0]": norm_action_dist[i][0],
            }
        elif algorithm == "ppo":
            aux_data = {
                "norm_action_dist[0]": norm_action_dist[i][0],
                "norm_action_dist[1]": norm_action_dist[i][1],
            }
        else:
            aux_data = {}

        exp = Experience(
            np.array(
                [norm_obs[0][0][i], norm_obs[1][0][i], norm_obs[2][0][i]],
                dtype=np.float32,
            ),
            np.array([norm_action[i][0]], dtype=np.float32),
            np.array(
                [
                    new_norm_obs[0][0][i],
                    new_norm_obs[1][0][i],
                    new_norm_obs[2][0][i],
                ],
                dtype=np.float32,
            ),
            float(reward[i][0]),
            False, # bool(done[i][0]),
            aux_data=aux_data,
        )
        experiences.append(exp)
    '''
    for i in range(1, len(norm_obs)):
        #print('Adding Aux data :')
        if algorithm == "ddpg":
            aux_data = {
                "norm_action1_dist[0]": norm_action_dist[0][0][i],
                "norm_action2_dist[0]": norm_action_dist[1][0][i],
                "norm_action3_dist[0]": norm_action_dist[2][0][i],
                "norm_action4_dist[0]": norm_action_dist[3][0][i],
                "IMEP_ref" : IMEP[i][0],
                "IMEP_actual": IMEP[i][1],
                "Rand_0": Rand_0,
                "Rand_1": Rand_1,
                "Rand_2": Rand_2,
                "Rand_3": Rand_3,
            }
        elif algorithm == "ppo":
            aux_data = {
                "norm_action_dist[0]": norm_action_dist[i][0],
                "norm_action_dist[1]": norm_action_dist[i][1],
            }
        else:
            aux_data = {}
        #print('Aux data is')
        #print(aux_data)
        exp = Experience(
            np.array(
                [norm_obs[i][0], norm_obs[i][1], norm_obs[i][2],
                 norm_obs[i][3], norm_obs[i][4], norm_obs[i][5],
                 norm_obs[i][6], norm_obs[i][7], norm_obs[i][8],
                 norm_obs[i][9], norm_obs[i][10], norm_obs[i][11],
                 norm_obs[i][12], norm_obs[i][13], norm_obs[i][14],
                 norm_obs[i][15]],
                dtype=np.float32,
            ),
            np.array([norm_action[0][0][i],norm_action[1][0][i],
                      norm_action[2][0][i],norm_action[3][0][i]], dtype=np.float32),
            np.array(
                [new_norm_obs[i][0], new_norm_obs[i][1], new_norm_obs[i][2],
                 new_norm_obs[i][3], new_norm_obs[i][4], new_norm_obs[i][5],
                 new_norm_obs[i][6], new_norm_obs[i][7], new_norm_obs[i][8],
                 new_norm_obs[i][9], new_norm_obs[i][10], new_norm_obs[i][11],
                 new_norm_obs[i][12], new_norm_obs[i][13], new_norm_obs[i][14],
                 new_norm_obs[i][15]],
                dtype=np.float32,
            ),
            float(reward[i][0]),
            False, # bool(done[i][0]),
            aux_data=aux_data,
        )
        experiences.append(exp)
    # Remove excess experiences
    idx = None
    for i, e in enumerate(experiences):
        if e.done:
            idx = i
            break
    if idx is not None:
        while len(experiences) > idx + 1:
            del experiences[-1]

    # Post-process experiences
    for e in experiences:
        denorm_obs = denormalize_obs(e.obs).tolist()
        denorm_action = denormalize_action(e.action).tolist()
        denorm_new_obs = denormalize_obs(e.new_obs).tolist()
        e.aux_data.update(
            {
                "denorm_obs": denorm_obs,
                "denorm_action": denorm_action,
                "denorm_new_obs": denorm_new_obs,
            }
        )
    #print(experiences)
    return Episode(agent_id, experiences)


def prepare_env(
    nn_bytes: bytes,
    cycle_no: int,
    minion_params: dict[str, Any],
    is_training_episode: bool,
) -> None:
    """Prepare the pendulum environment in Simulink for a training or a
    validation run.

    Arguments:
        - nn_bytes: bytes
              Bytes of the TensorFlow Lite model for the agent's behavior.
        - cycle_no: int
              Current LExCI cycle number.
        - minion_params: dict[str, Any]
              Miscellaneous parameters.
        - is_training_episode: bool
              Whether to prepare the environment for a training run (`True`) or
              a validation run (`False`).
    """

    # Clear the buffer
    controller.run_cmd("clear experience_buffer")

    # Overwrite the RL Block's adjustable parameters
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/norm_observation_size",
        16,
    )
    if algorithm == "ddpg":
        controller.write_simulink_var(
            "H2DFRL_LExCI_env/RL_Block/RL_Agent/norm_action_dist_size",
            4,
        )
    elif algorithm == "ppo":
        controller.write_simulink_var(
            "H2DFRL_LExCI_env/RL_Block/RL_Agent/norm_action_dist_size",
            8,
        )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/static_nn_memory",
        f"uint8({list(nn_bytes)})",
    )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/tensor_arena_size",
        100000,
    )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/lower_observation_bounds",
        [-20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,-20, -20, -20, -20],
    )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/upper_observation_bounds",
        [+20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20, +20],
    )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/lower_action_bounds",
        [-2.3201,-2.1257,-1.8309, -1.6562],
    )
    controller.write_simulink_var(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/upper_action_bounds",
        [+1.4113, 0.9867,0.9542, 2.1874],
    )
    if is_training_episode:
        controller.write_simulink_var(
            "H2DFRL_LExCI_env/RL_Block/RL_Agent/b_training_active",
            1,
        )
    else:
        controller.write_simulink_var(
            "H2DFRL_LExCI_env/RL_Block/RL_Agent/b_training_active",
            0,
        )

    # Set the standard deviation for DDPG's sampling system
    if algorithm == "ddpg":
        initial_stdev = minion_params.get("INITIAL_STDEV", 0.5)
        stdev_decay_factor = minion_params.get("STDEV_DECAY_FACTOR", 0.925)
        controller.write_simulink_var(
            "H2DFRL_LExCI_env/RL_Block/RL_Agent/ddpg_standard_deviation",
            calc_stdev(initial_stdev, stdev_decay_factor, cycle_no),
        )

    # Set the seeds of the RNGs
    controller.write_simulink_block_param(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/Sampling/Continuous Action/DDPG_Sampler/Random\nNumber",
        "Seed",
        int(time.time()),
    )
    controller.write_simulink_block_param(
        "H2DFRL_LExCI_env/RL_Block/RL_Agent/Sampling/Continuous Action/PPO_Sampler/Random\nNumber",
        "Seed",
        int(time.time()),
    )

    # Set the initial values
    if is_training_episode:

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND0",
            np.random.uniform(4e5,10e5),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND1",
            np.random.uniform(0.5,1.3),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND2",
            np.random.uniform(0.75,1.4),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND3",
            np.random.uniform(0.8,1.2),
        )
  
    else:
        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND0",
            np.random.uniform(4e5,10e5),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND1",
            np.random.uniform(0.5,1.3),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND2",
            np.random.uniform(0.75,1.4),
        )

        controller.write_simulink_var(
            r"H2DFRL_LExCI_env/RAND3",
            np.random.uniform(0.8,1.2),
        )


def generate_training_data(
    model_bytes: bytes,
    cycle_no: int,
    num_experiences: int,
    minion_params: dict[str, Any],
) -> Cycle:
    """Generate training data.

    Arguments:
      - model_bytes: bytes
          Bytes of the TensorFlow Lite model for the agent's behavior.
      - cycle_no: int
          Current LExCI cycle number.
      - num_experiences: int
          Number of experiences to generate.
      - minion_params: dict[str, Any]
          Miscellaneous parameters.

    Returns:
      - _: Cycle
          `Cycle` object containing the generated data.
    """

    num_collected_exps = 0
    num_episodes = 1
    cycle = Cycle()

    print(f"========== Training Cycle {cycle_no} ==========")
    while num_collected_exps < num_experiences:
        print(f"Starting episode {num_episodes}.")

        # Prepare the environment
        prepare_env(model_bytes, cycle_no, minion_params, True)

        # Start the simulation and wait for it to end
        controller.run_cmd("experience_buffer = sim('H2DFRL_LExCI_env');")

        # Retrieve the data
        episode = retrieve_episode(controller, "agent0")
        cycle.add_episode(episode)
        num_collected_exps += len(episode)
        num_episodes += 1

    return cycle


def generate_validation_data(
    model_bytes: bytes, cycle_no: int, minion_params: dict[str, Any]
) -> Cycle:
    """Generate training data.

    Arguments:
      - model_bytes: bytes
          Bytes of the TensorFlow Lite model for the agent's behavior.
      - cycle_no: int
          Current LExCI cycle number.
      - minion_params: dict[str, Any]
          Miscellaneous parameters.

    Returns:
      - _: Cycle
          `Cycle` object containing the generated data.
    """

    print(f"========== Validation Cycle {cycle_no} ==========")

    # Prepare the environment
    prepare_env(model_bytes, cycle_no, minion_params, False)

    # Start the simulation and wait for it to end
    controller.run_cmd("experience_buffer = sim('H2DFRL_LExCI_env');")

    # Retrieve the data
    episode = retrieve_episode(controller, "agent0")
    cycle = Cycle()
    cycle.add_episode(episode)
    return cycle


if __name__ == "__main__":
    minion = Minion(
        "127.0.0.1", 5555, generate_training_data, generate_validation_data
    )
    controller.start_matlab_simulink(
        "/home/vasu3/Documents/work/H2DFRL_LExCI/",
        "H2DFRL_LExCI_env",
        "ReinforcementLearningBlockInit_H2DFRL",
        False,
    )
    controller.run_cmd('init_H2DFRL_GRU')
    minion.mainloop()
    controller.stop_matlab_simulink()
