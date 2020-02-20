import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.BCELoss()
GC = False 

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
            super(ForwardModel, self).__init__()

            self.l1 = nn.Linear(state_dim + 1, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, state_dim)
            

    def forward(self, state, q):
            sa = torch.cat([state, q], 1)
            next_state = F.relu(self.l1(sa))
            next_state = F.relu(self.l2(next_state))
            return self.l3(next_state)

class Model(nn.Module):
        def __init__(self, state_dim):
                super(Model, self).__init__()

                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, state_dim)

        def forward(self, state):
                next_state = F.relu(self.l1(state))
                next_state = F.relu(self.l2(next_state))
                return self.l3(next_state)

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


class Critic(nn.Module):
        def __init__(self, state_dim):
                super(Critic, self).__init__()

                # Q1 architecture
                self.l1 = nn.Linear(2 * state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 1)

                # Q2 architecture
                self.l4 = nn.Linear(2 * state_dim, 256)
                self.l5 = nn.Linear(256, 256)
                self.l6 = nn.Linear(256, 1)

        def forward(self, state, next_state):
                ss = torch.cat([state, next_state], 1)

                q1 = F.relu(self.l1(ss))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)

                q2 = F.relu(self.l4(ss))
                q2 = F.relu(self.l5(q2))
                q2 = self.l6(q2)
                return q1, q2


        def Q1(self, state, next_state):
                ss = torch.cat([state, next_state], 1)

                q1 = F.relu(self.l1(ss))
                q1 = F.relu(self.l2(q1))
                q1 = self.l3(q1)

                return q1


class D3G(object):
        def __init__(
                self,
                state_dim,
                action_dim,
                max_action,
                summary_name,
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                batch_size=100
        ):

                self.actor = Actor(state_dim, action_dim, max_action).to(device)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
                self.actor_target = copy.deepcopy(self.actor)

                self.critic = Critic(state_dim).to(device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

                self.model = Model(state_dim).to(device)
                self.model_target = copy.deepcopy(self.model)
                self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

                self.forward_model = ForwardModel(state_dim, action_dim).to(device)
                self.forward_model_target = copy.deepcopy(self.forward_model)
                self.forward_model_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=3e-4)

                self.max_action = max_action
                self.discount = discount
                self.tau = tau
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                self.batch_size = batch_size

                self.total_it = 0

        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.model(state).detach()

                return self.actor(state, next_state).cpu().data.numpy().flatten()

        def select_goal(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.model(state).detach()
                inverse_q = self.critic_target.Q1(state, next_state)
                next_state = state + self.forward_model(state, inverse_q)

                return next_state.cpu().data.numpy().flatten()

        def distance(self, state, goal):
            return 1. / sum((state - goal) ** 2)

        def train_actor(self, replay_buffer):
                state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

                # Compute actor loss 
                predicted_action = self.actor(state, next_state)
                actor_loss = F.mse_loss(predicted_action, action)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()



        def train(self, replay_buffer, dynamics_only=False):
                # Sample replay buffer 
                state, action, next_state, goal_state, reward, _, not_done = replay_buffer.sample(self.batch_size)
                _, _, not_next_state, _, _, _, _ = replay_buffer.sample(self.batch_size)

                if not dynamics_only:
                    self.total_it += 1

                    ##################################################### Q(s, s') computation ######################################################################
                    with torch.no_grad():
                            # Select action according to policy and add clipped noise
                            next_next_state = next_state + self.model_target(next_state)
                            target_Q1 = self.critic_target.Q1(next_state, next_next_state)
                            cycle_next_next_state = next_state + self.forward_model(next_state, target_Q1)

                            # Compute the target Q value
                            target_Q1, target_Q2 = self.critic_target(next_state, cycle_next_next_state)
                            target_Q = torch.min(target_Q1, target_Q2)
                            target_Q = reward + not_done * self.discount * target_Q

                    # Get current Q estimates
                    current_Q1, current_Q2 = self.critic(state, next_state)

                    # Compute critic loss
                    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                   
                    # Optimize the critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                # Compute forward model loss
                #predicted_next_state = self.forward_model(state, reward)
                target_Q1 = self.critic_target.Q1(state, next_state)
                predicted_next_state = self.forward_model(state, target_Q1.detach())
                delta = next_state - state
                forward_model_loss = F.mse_loss(delta, predicted_next_state) 

                # Optimize the forward model
                self.forward_model_optimizer.zero_grad()
                forward_model_loss.backward()
                self.forward_model_optimizer.step()

                # Delayed model updates
                if self.total_it % self.policy_freq == 0 and not dynamics_only:
                    # Compute model loss
                    model_prediction = state + self.model(state)
                    inverse_q = self.critic.Q1(state, model_prediction)
                    cycle_next_state = state + self.forward_model(state, inverse_q)
                    model_gradient_loss = -self.critic.Q1(state, cycle_next_state).mean() 

                    cycle_loss = F.mse_loss(cycle_next_state, model_prediction)
                    model_loss = model_gradient_loss + cycle_loss
                    # Optimize the model 
                    self.model_optimizer.zero_grad()
                    model_loss.backward()
                    self.model_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

