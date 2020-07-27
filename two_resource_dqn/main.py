import configparser

import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from two_resource_dqn.experiment import Experiment


def show_config(config: configparser.ConfigParser):
    for name, section in config.items():
        print(f"---- {name}")
        for k, v in section.items():
            print(f"{k} = {v}")


def main(config):
    env_name = "./unity_env/TwoResourceProblem"
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           seed=1,
                           side_channels=[engine_configuration_channel])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    experiment = Experiment(env=env, config=config, device=device)
    experiment.start()
    env.close()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("./config.ini")
    show_config(config)
    main(config)
