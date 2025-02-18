"""Minion for the FOR2401 project.

File:   for2401_minion.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-11-08
"""

import lexci2
from lexci2.minion.minion import Minion
from lexci2.neural_network_modules.continuous_ppo_neural_network_module import (
    ContinuousPpoNeuralNetworkModule,
)
from lexci2.agents.ppo_agent import PpoAgent
from lexci2.data_containers import Experience, Episode, Cycle
from mabx_network_interface import MabxNetworkInterface
from lexci2.lexci_env import LexciEnvConfig


import logging
import datetime
import numpy as np
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger()


# Network interface to the MABX
mni = MabxNetworkInterface("192.168.140.110", 42787, "192.168.140.159", 42787)

# TODO: Modify the lower/upper action bound.
env_config = LexciEnvConfig(
    16,
    4,
    "continuous",
    obs_lb=np.array([-20, -20, -20, -20, -20, -20, -20, -20,-20, -20, -20, -20, -20, -20, -20, -20], dtype=np.float32),
    obs_ub=np.array([+20, +20, +20, +20, +20, +20, +20, +20,+20, +20, +20, +20, +20, +20, +20, +20], dtype=np.float32),
    action_lb=np.array([-2.3201   -2.1257   -0.2394   -1.2291], dtype=np.float32),
    action_ub=np.array([1.4113,    0.9867,    0.1585,    2.1874], dtype=np.float32),
    norm_obs_lb=np.array(16 * [-1], dtype=np.float32),
    norm_obs_ub=np.array(16 * [1], dtype=np.float32),
    norm_action_lb=np.array(4 * [-np.inf], dtype=np.float32),
    norm_action_ub=np.array(4 * [+np.inf], dtype=np.float32),
)


def calc_stdev(
    initial_stdev: float, decay_factor: float, cycle_no: int
) -> float:
    """Calculate the standard deviation of the action distribution as a function
    of the current LExCI cycle.

    Arguments:
      - initial_stdev: float
          Initial value of the standard deviation.
      - decay_factor: float
          Decay factor.
      - cycle_no: int
          Current LExCI cycle number.

    Returns:
      - _: float
          The standard deviation of the action distribution.
    """

    return initial_stdev * decay_factor**cycle_no


def generate_training_data(
    nn_bytes: bytes,
    cycle_no: int,
    num_experiences: int,
    minion_params: dict[str, Any],
) -> Cycle:
    """Generate training data.

    Arguments:
      - nn_bytes: bytes
          Bytes of the agent's neural network.
      - cycle_no: int
          Current LExCI cycle number.
      - num_experiences: int
          Number of experiences to generate.
      - minion_params: dict[str, Any]
          Dictionary with parameters from the Master.

    Returns:
      - _: Cycle
          Cycle containing the generated data.
    """

    nnm = ContinuousPpoNeuralNetworkModule(
        env_config,
        nn_bytes,
        nn_data_fmt="tflite"
       )
    cycle = Cycle()


    while cycle.get_num_experiences() < num_experiences:
        episode = Episode("ppo_agent0")

        # Wait until the MABX signals that it's ready to start the episode
        while True:
            d = mni.recv()
            if d["bMABX Ready"]:
                break

        # Prompt the MABX to start a new episode
        d = {
            "DOIMain": 0.0,
            "P2M": 0.0,
            "SOIMain": 0.0,
            "DOIH2": 0.0,
            "PiCycleCounter": 0,  # TODO: Align with Julian
            "bStrtReq": True,
            "bValidationCycle": False,
        }
        mni.send(d)
        # Run the episode
        prev_norm_obs = None
        prev_norm_action = None
        t_start = datetime.datetime.now()
        b_terminate_episode = False
        c = 0
        while not b_terminate_episode:
            # Receive the current observation and compute an action
            d = mni.recv()
            b_terminate_episode = d["bTerminateEpisode"]
            Op_State = d["OpState"][0]

            norm_obs = np.array(
			[
			    d["IMEPLast"],
			    d["NOXLast"],
			    d["DeltaNOx"],
			    d["MPRR"],
			    d["IMEPRefLast"],
			    d["IMEPRef"],
			    d["DeltaIMEP"],
			    d["ErrorIMEP"],
			    d["HiddenState1"],
			    d["HiddenState2"],
			    d["HiddenState3"],
			    d["HiddenState4"],
			    d["HiddenState5"],
			    d["HiddenState6"],
			    d["HiddenState7"],
			    d["HiddenState8"],
			],
                dtype=np.float32,
            )
            norm_action = nnm.get_norm_action(norm_obs, True)
            #norm_action = np.array([0,0,0,0], dtype = np.float32)

            # Save the previous experience
            if prev_norm_obs is not None and prev_norm_action is not None:
                t = (datetime.datetime.now() - t_start).total_seconds()
                aux_data = {"bTerminateEpisode": b_terminate_episode,
                        "OpState": Op_State}
                if Op_State != 2 or b_terminate_episode is True:
                    experience = Experience(
                        prev_norm_obs,
                        prev_norm_action,
                        norm_obs,
                        d["Reward"][0],
                        d["bTerminateEpisode"],
                        t,
                        aux_data = aux_data
                    )
                    episode.append_experience(experience)
            prev_norm_obs = norm_obs
            prev_norm_action = norm_action

            # Send the action back to the MABX
            d = {
                "DOIMain": norm_action[0],
                "P2M": norm_action[1],
                "SOIMain": norm_action[2],
                "DOIH2": norm_action[3],
                "PiCycleCounter": 0,  # TODO: Align with Julian
                "bStrtReq": False,
                "bValidationCycle": False,
            }
            mni.send(d)
        if len(episode)>0:
            cycle.add_episode(episode)
    print(cycle)
    return cycle


def generate_validation_data(
    nn_bytes: bytes, cycle_no: int, minion_params: dict[str, Any]
) -> Cycle:
    """Generate validation data.

    Arguments:
      - nn_bytes: bytes
          Bytes of the agent's neural network.
      - cycle_no: int
          Current LExCI cycle number.
      - minion_params: dict[str, Any]
          Dictionary with parameters from the Master.

    Returns:
      - _: Cycle
          Cycle containing the validation data.
    """

    nnm = ContinuousPpoNeuralNetworkModule(
        env_config,
        nn_bytes,
        nn_data_fmt="tflite",
        )

    cycle = Cycle()

    episode = Episode("ppo_agent0")


    # Wait until the MABX signals that it's ready to start the episode
    while True:
        d = mni.recv()

        if d["bMABX Ready"]:
            print('MABX ready')
            break

    # Prompt the MABX to start a new episode
    d = {
        "DOIMain": 0.0,
        "P2M": 0.0,
        "SOIMain": 0.0,
        "DOIH2": 0.0,
        "PiCycleCounter": 0,  # TODO: Align with Julian
        "bStrtReq": True,
        "bValidationCycle": True,
    }
    

    mni.send(d)


    # Run the episode
    prev_norm_obs = None
    prev_norm_action = None
    t_start = datetime.datetime.now()
    b_terminate_episode = False
    while not b_terminate_episode:
        # Receive the current observation and compute an action
        d = mni.recv()
        b_terminate_episode = d["bTerminateEpisode"]
        Op_State = d["OpState"][0]

        norm_obs = np.array(
			[
			    d["IMEPLast"],
			    d["NOXLast"],
			    d["DeltaNOx"],
			    d["MPRR"],
			    d["IMEPRefLast"],
			    d["IMEPRef"],
			    d["DeltaIMEP"],
			    d["ErrorIMEP"],
			    d["HiddenState1"],
			    d["HiddenState2"],
			    d["HiddenState3"],
			    d["HiddenState4"],
			    d["HiddenState5"],
			    d["HiddenState6"],
			    d["HiddenState7"],
			    d["HiddenState8"],
			],
            dtype=np.float32,
        )
        norm_action = nnm.get_norm_action(norm_obs, False)
        # Save the previous experience
        if prev_norm_obs is not None and prev_norm_action is not None:
            t = (datetime.datetime.now() - t_start).total_seconds()
            aux_data = {"bTerminateEpisode": b_terminate_episode,
                        "OpState": Op_State}
            if (Op_State != 2 and Op_State !=1) or b_terminate_episode is True:
                experience = Experience(
                    prev_norm_obs,
                    prev_norm_action,
                    norm_obs,
                    d["Reward"][0],
                    d["bTerminateEpisode"],
                    t,
                    aux_data = aux_data
                )
                episode.append_experience(experience)
        prev_norm_obs = norm_obs
        prev_norm_action = norm_action

        # Send the action back to the MABX
        d = {
                "DOIMain": norm_action[0],
                "P2M":  norm_action[1],
                "SOIMain":  norm_action[2],
                "DOIH2":  norm_action[3],
                "PiCycleCounter": 0,  # TODO: Align with Julian
                "bStrtReq": False,
                "bValidationCycle": True,
            }
        mni.send(d)
    cycle.add_episode(episode)
    return cycle



minion = Minion(
    "127.0.0.1", 5555, generate_training_data, generate_validation_data
)
minion.mainloop()
mni.terminate()
