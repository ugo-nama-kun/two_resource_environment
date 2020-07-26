import random

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, n_images, vector_dim, n_action):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3 * n_images, out_channels=32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc_vec = nn.Linear(in_features=vector_dim, out_features=256)
        self.fc1 = nn.Linear(in_features=5664, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=n_action)

        self.act_conv = nn.Softplus()
        self.act_fc = nn.Softplus()

    def forward(self, image_tensor, vector_tensor):
        x_im = self.act_conv(self.bn1(self.conv1(image_tensor)))
        x_im = self.act_conv(self.bn2(self.conv2(x_im)))
        x_im = x_im.flatten()
        x_int = self.act_fc(self.fc_vec(vector_tensor))
        x = torch.cat([x_im, x_int], dim=0)
        x = self.act_fc(self.fc1(x))
        output = self.fc2(x)
        return output


class DQNAgent:
    """ Classical Deep Q Network Agent
    """
    def __init__(self, config, n_action, action_size, shape_vector_obs, shape_obs_image):
        self.learning_rate = float(config["dnn"]["learning_rate"])
        self.adam_eps = float(config["dnn"]["adam_eps"])
        self.batch_size = int(config["dnn"]["batch_size"])
        self.replay_buffer = int(config["qn"]["replay_buffer"])
        self.iteration = int(config["qn"]["iteration"])
        self.exploration = float(config["qn"]["exploration"])
        self.input_time_horizon = int(config["qn"]["input_time_horizon"])
        self.action_size = action_size
        self.shape_obs_image = shape_obs_image  # Like (64, 64, 3)
        self.shape_vector_obs = shape_vector_obs  # Like  (2,)
        self.n_action = n_action

        self.qnet = QNet(n_images=self.input_time_horizon,
                         vector_dim=2,
                         n_action=n_action)
        self.qnet_support = QNet(n_images=1,
                                 vector_dim=2,
                                 n_action=n_action)
        self.qnet_support.load_state_dict(self.qnet.state_dict())
        self.time_tick = 0

    def step(self, observation, done) -> torch.Tensor:
        # plt.imshow(observation[0])
        # plt.pause(0.0001)
        # print(f"Vector observations : {observation[1]}")

        im_tensor, vec_tensor = self.obs_to_tensor(observation)
        q_val = self.qnet(im_tensor, vec_tensor)
        print(q_val)

        # Copy Q net to the support network
        if self.time_tick == self.iteration:
            self.time_tick = 0
            self.qnet_support.load_state_dict(self.qnet.state_dict())
        self.time_tick += 1
        return random.choice(range(self.n_action))

    @staticmethod
    def obs_to_tensor(raw_observation):
        im_tensor = torch.tensor(raw_observation[0]).view((1, 3, 64, 64))
        vec_tensor = torch.tensor(raw_observation[1])
        return im_tensor, vec_tensor
