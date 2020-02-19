"""Behavioral Cloning from Observation"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
                super(Actor, self).__init__()

                self.l1 = nn.Linear(2 * state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                self.max_action = max_action

        def forward(self, state, next_state):
                ss = torch.cat([state, next_state], 1)
                a = F.relu(self.l1(ss))
                a = F.relu(self.l2(a))

                return self.max_action * torch.tanh(self.l3(a))

class BC(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
                super(BC, self).__init__()

                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                self.max_action = max_action

        def forward(self, state):
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))

                return self.max_action * torch.tanh(self.l3(a))

class BCO(object):
        def __init__(self, state_dim, action_dim, max_action, batch_size=100):
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

            self.bc = BC(state_dim, action_dim, max_action).to(device)
            self.bc_optimizer = torch.optim.Adam(self.bc.parameters(), lr=3e-4)

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.batch_size=batch_size

        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                return self.bc(state).cpu().data.numpy().flatten()

        def train_actor(self, replay_buffer):
            states, actions, next_states, _, _ = replay_buffer.sample(self.batch_size)
            predicted_actions = self.actor(states, next_states)
            loss = F.mse_loss(predicted_actions, actions)
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

        def train_bc(self, observational_buffer):
            states, next_states = observational_buffer.sample(self.batch_size)
            actions = self.actor(states, next_states).detach()
            predicted_actions = self.bc(states)

            loss = F.mse_loss(predicted_actions, actions)
            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()
