import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from configparser import ConfigParser
from mlagents_envs.base_env import TerminalSteps, DecisionSteps
from mlagents_envs.environment import UnityEnvironment
from two_resource_dqn.dqn_agent import DQNAgent


class Experiment:
    """ Experiment Class for handling the experiment process
    Assuming only one agent (behavior) is in the environment
    """

    def __init__(self,
                 env: UnityEnvironment,
                 config: ConfigParser):
        self.env = env
        self.config = config
        self.n_experiment = int(config["experiment"]["n_experiment"])
        self.n_episode = int(config["experiment"]["n_episode"])
        self.maximum_survival_time_steps = int(config["experiment"]["maximum_survival_time_steps"])

        # Performance
        self.hist_survival_time_steps = []
        sns.set_style(style="darkgrid")

        # Initialization of the environment and the agent
        self.env.reset()
        print("---- Agent Specs")
        self.behavior_name = list(self.env.behavior_specs)[0]
        print(f"Name of the behavior : {self.behavior_name}")
        self.spec = self.env.behavior_specs[self.behavior_name]
        self.dqn_agent = DQNAgent(n_action=self.spec.discrete_action_branches[0],
                                  action_size=self.spec.action_size,
                                  shape_vector_obs=self.spec.observation_shapes[1],
                                  shape_obs_image=self.spec.observation_shapes[0])
        print(f"n_actions : {self.dqn_agent.n_action}")
        print(f"Shape of the Vector Observation : {self.dqn_agent.shape_vector_obs}")
        print(f"Shape of the Image Observation : {self.dqn_agent.shape_obs_image}")
        print("----")

    def start(self):
        fig = plt.figure()
        line_props = {"linestyle": "--", "color": "k"}
        plt.plot([self.maximum_survival_time_steps] * self.n_episode, **line_props)
        for n in range(self.n_experiment):
            survival_time_steps = []
            for episode in range(self.n_episode):
                done = False
                t = 0
                self.env.reset()
                while not done:
                    decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                    self.decision_process(decision_steps)
                    done = self.terminal_process(terminal_steps)
                    if done or t == self.maximum_survival_time_steps:
                        survival_time_steps.append(t)
                    else:
                        t += 1
                print(f"{n}/{self.n_experiment}-th experiment, episodes: {episode}/{self.n_episode}, Score: {t} ")
            plt.plot(range(self.n_episode), survival_time_steps)
            plt.pause(0.001)

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
            self.env.set_actions(self.behavior_name, action)
            self.env.step()

    def terminal_process(self, terminal_steps: TerminalSteps):
        # Terminal steps returns the ids of agents at the terminal
        done = False
        for agent_id_terminated in terminal_steps:
            image = terminal_steps[agent_id_terminated].obs[0]
            vector_obs = terminal_steps[agent_id_terminated].obs[1]
            done = True
            self.dqn_agent.step(observation=(image, vector_obs), done=done)
        return done
