import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Dynamics Gradients (D3G)

class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
            super(ForwardModel, self).__init__()

            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, state_dim)
            

    def forward(self, state, action):
            sa = torch.cat([state, action], 1)
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


class D3G(object):
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

                self.forward_model = ForwardModel(state_dim, action_dim).to(device)
                self.forward_model_target = copy.deepcopy(self.forward_model)
                self.forward_model_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=3e-4)

                self.max_action = max_action
                self.is_discrete = is_discrete
                self.discount = discount
                self.tau = tau
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                
                self.total_it = 0

        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.model(state).detach()
                action = self.actor(state, next_state).cpu().data.numpy().flatten()

                if self.is_discrete:
                    action = np.argmax(action)

                return action

        def select_goal(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                next_state = state + self.model(state).detach()
                inverse_a = self.actor(state, next_state)
                next_state = state + self.forward_model(state, inverse_a)

                return next_state.cpu().data.numpy().flatten()

        def train(self, replay_buffer, batch_size=100, dynamics_only=False):
                # Sample replay buffer 
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                if not dynamics_only:
                    self.total_it += 1
                    with torch.no_grad():
                            # Select action according to policy and add clipped noise
                            next_next_state = next_state + self.model_target(next_state)
                            inverse_action = self.actor(next_state, next_next_state)
                            cycle_next_next_state = next_state + self.forward_model(next_state, inverse_action)

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

                # Compute actor loss 
                predicted_action = self.actor(state, next_state)
                actor_loss = F.mse_loss(predicted_action, action)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Compute forward model loss
                predicted_next_state = self.forward_model(state, action)
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
                    inverse_action = self.actor(state, model_prediction)
                    cycle_next_state = state + self.forward_model(state, inverse_action)

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


        def save(self, filename):
                torch.save(self.critic.state_dict(), filename + "_critic")
                torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                torch.save(self.actor.state_dict(), filename + "_actor")
                torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
                torch.save(self.model.state_dict(), filename + "_model")
                torch.save(self.model_optimizer.state_dict(), filename + "_model_optimizer")
                torch.save(self.forward_model.state_dict(), filename + "_forward_model")
                torch.save(self.forward_model_optimizer.state_dict(), filename + "_forward_model_optimizer")

        def load(self, filename):
                self.critic.load_state_dict(torch.load(filename + "_critic"))
                self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
                self.actor.load_state_dict(torch.load(filename + "_actor"))
                self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
                self.model.load_state_dict(torch.load(filename + "_model"))
                self.model_optimizer.load_state_dict(torch.load(filename + "_model_optimizer"))
                self.forward_model.load_state_dict(torch.load(filename + "_forward_model"))
                self.forward_model_optimizer.load_state_dict(torch.load(filename + "_forward_model_optimizer"))
                self.model_target = copy.deepcopy(self.model)
                self.critic_target = copy.deepcopy(self.critic)
