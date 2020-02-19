import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

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
        def __init__(self, state_dim, action_dim, max_action, is_discrete):
                super(Actor, self).__init__()

                self.l1 = nn.Linear(2 * state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, action_dim)
                self.max_action = max_action
                self.is_discrete = is_discrete
                

        def forward(self, state, next_state):
                ss = torch.cat([state, next_state], 1)
                a = F.relu(self.l1(ss))
                a = F.relu(self.l2(a))

                if self.is_discrete:
                    return torch.nn.Softmax()(self.l3(a))
                else:
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


class Standard_QSS(object):
        def __init__(
                self,
                state_dim,
                action_dim,
                max_action,
                is_discrete=False,
                discount=0.99,
                tau=0.005,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2
        ):

                self.actor = Actor(state_dim, action_dim, max_action, is_discrete).to(device)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

                self.critic = Critic(state_dim).to(device)
                self.critic_target = copy.deepcopy(self.critic)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

                self.model = Model(state_dim).to(device)
                self.model_target = copy.deepcopy(self.model)
                self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

                self.max_action = max_action
                self.is_discrete = is_discrete
                self.discount = discount
                self.tau = tau
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                
                self.total_it = 0
                self.writer = SummaryWriter("loss_summaries")


        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.model(state).detach()
                action = self.actor(state, next_state).cpu().data.numpy().flatten()

                if self.is_discrete:
                    action = np.argmax(action)

                return action


        def train(self, replay_buffer, batch_size=100, dynamics_only=False):
                # Sample replay buffer 
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                if not dynamics_only:
                    self.total_it += 1
                    with torch.no_grad():
                            # Select action according to policy and add clipped noise
                            next_next_state = next_state + self.model_target(next_state)

                            # Compute the target Q value
                            target_Q1, target_Q2 = self.critic_target(next_state, next_next_state)
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

                # Compute actor loss 
                predicted_action = self.actor(state, next_state)
                actor_loss = F.mse_loss(predicted_action, action)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Delayed model updates
                if self.total_it % self.policy_freq == 0 and not dynamics_only:
                    # Compute model loss
                    model_prediction = state + self.model(state)
                    model_loss = -self.critic.Q1(state, model_prediction).mean() 
                    
                    # Optimize the model 
                    self.model_optimizer.zero_grad()
                    model_loss.backward()
                    self.model_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                  
                    self.writer.add_scalar("Loss/model_loss", model_loss, self.total_it)
                    self.writer.add_scalar("Loss/actor_loss", actor_loss, self.total_it)
                    self.writer.add_scalar("Loss/critic_loss", critic_loss, self.total_it)
                    self.writer.add_scalar("Loss/model_loss", model_loss, self.total_it)
                      

        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                torch.save(self.actor.state_dict(), filename + "_actor")
                torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
                torch.save(self.model.state_dict(), filename + "_model")
                torch.save(self.model_optimizer.state_dict(), filename + "_model_optimizer")

        def load(self, filename):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.actor.load_state_dict(torch.load(filename + "_actor"))
                self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
                self.model.load_state_dict(torch.load(filename + "_model"))
                self.model_optimizer.load_state_dict(torch.load(filename + "_model_optimizer"))
                self.actor_target = copy.deepcopy(self.actor)
                self.critic_target = copy.deepcopy(self.critic)
