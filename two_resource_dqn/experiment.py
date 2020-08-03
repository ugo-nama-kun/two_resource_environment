import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from configparser import ConfigParser
from mlagents_envs.base_env import TerminalSteps, DecisionSteps
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from two_resource_dqn.dqn_agent import DQNAgent


class Experiment:
    """ Experiment Class for handling the experiment process
    Assuming only one agent (behavior) is in the environment
    """

    def __init__(self,
                 env: UnityEnvironment,
                 env_channel: EngineConfigurationChannel,
                 config: ConfigParser,
                 device):
        self.env = env
        self.env_channel = env_channel
        self._config = config
        self._n_experiment = int(config["experiment"]["n_experiment"])
        self._n_episode = int(config["experiment"]["n_episode"])
        self._maximum_survival_time_steps = int(config["experiment"]["maximum_survival_time_steps"])
        self._device = device
        self._save_network_every = int(config["experiment"]["save_network_every"])

        # Exploration Scheduling
        self._training_start_from = int(config["qn"]["training_start_from"])
        self._eps_start = float(config["exploration"]["eps_start"])
        self._eps = self._eps_start
        self._eps_min = float(config["exploration"]["eps_min"])
        self._eps_delta = float(config["exploration"]["eps_delta"])
        self._eps_checkpoints = [int(v) for v in config["exploration"]["eps_checkpoints"].split(".")]

        # Performance
        self._hist_survival_time_steps = []
        sns.set_style(style="darkgrid")

        # Initialization of the environment and the agent
        self.env.reset()
        print("---- Agent Specs")
        self._behavior_name = list(self.env.behavior_specs)[0]
        self._spec = self.env.behavior_specs[self._behavior_name]
        print(f"Name of the behavior : {self._behavior_name}")
        self.dqn_agent = None
        self.init_agent_params()
        print("----")

    def init_agent_params(self):
        self._eps = self._eps_start
        print(f"Initial exploration : {self._eps}")
        self.dqn_agent = DQNAgent(config=self._config,
                                  n_action=self._spec.discrete_action_branches[0],
                                  action_size=self._spec.action_size,
                                  shape_vector_obs=self._spec.observation_shapes[1],
                                  shape_obs_image=self._spec.observation_shapes[0],
                                  eps_start=self._eps_start,
                                  device=self._device)
        print(f"n_actions : {self.dqn_agent.n_action}")
        print(f"Shape of the Vector Observation : {self.dqn_agent.shape_vector_obs}")
        print(f"Shape of the Image Observation : {self.dqn_agent.shape_obs_image}")

    def start(self):
        fig = plt.figure()
        line_props = {"linestyle": "-", "color": "r"}
        line, = plt.plot(range(self._n_episode), [0] * self._n_episode, **line_props)
        plt.xlim([0, self._n_episode])
        plt.ylim([0, self._maximum_survival_time_steps])
        plt.pause(0.01)
        for n in range(self._n_experiment):
            survival_time_steps = [0] * self._n_episode
            self.init_agent_params()
            for episode in range(self._n_episode):
                t = 0
                done = False
                self.update_eps_scheduled(episode=episode,
                                          time_step=sum([v for v in survival_time_steps if v is not None]))
                self.env.reset()
                self.dqn_agent.reset_for_new_episode()
                while not done:
                    print(f"experiment : {n}/{self._n_experiment}, episode : {episode}/{self._n_episode}, exploration: {self._eps}")
                    decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
                    self.decision_process(decision_steps)
                    done = self.terminal_process(terminal_steps)
                    if done or t >= self._maximum_survival_time_steps:
                        print("Episode Done.")
                        survival_time_steps[episode] = t
                        line.set_data(range(self._n_episode), survival_time_steps)
                        plt.pause(0.01)
                    else:
                        t += 1
                print(f"{n}/{self._n_experiment}-th experiment, episodes: {episode}/{self._n_episode}, Score: {t} ")
                if episode + 1 % self._save_network_every == 0:
                    self.dqn_agent.save_network(n_experiment=n)
            plt.plot(range(self._n_episode), survival_time_steps)
            plt.pause(0.01)
            self.dqn_agent.save_network(n_experiment=n)

        print("Experiment Done.")
        plt.show()

    def decision_process(self, decision_steps: DecisionSteps):
        # Decision steps returns the ids of the action request of agents
        for agent_id_decision in decision_steps:
            image = decision_steps[agent_id_decision].obs[0]
            vector_obs = decision_steps[agent_id_decision].obs[1]

            # action = spec.create_random_action(len(decision_steps))
            action = self.dqn_agent.step(observation=(image, vector_obs), done=False)
            action = np.array([[action]])

            # Move the simulation forward
            self.env.set_actions(self._behavior_name, action)
            self.env.step()

    def terminal_process(self, terminal_steps: TerminalSteps):
        # Terminal steps returns the ids of agents at the terminal
        done = False
        for agent_id_terminated in terminal_steps:
            # image = terminal_steps[agent_id_terminated].obs[0]
            # vector_obs = terminal_steps[agent_id_terminated].obs[1]
            done = True
            # self.dqn_agent.step(observation=(image, vector_obs), done=done)
        return done

    def update_eps_scheduled(self, episode,  time_step):
        if time_step < self._training_start_from:
            self.dqn_agent.eps_e_greedy = 1.0
            return

        if episode in self._eps_checkpoints:
            self._eps -= self._eps_delta
            if self._eps < self._eps_min:
                self._eps = self._eps_min
            print(f"Exploration parameter of e-greedy was updated to {self._eps}")
        self.dqn_agent.eps_e_greedy = self._eps

