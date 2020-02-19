import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import ObservationalD3G
import bco
from collections import deque 
import cv2 
import random 
import pickle


def done_condition(env_name, state):
    if 'Reacher' in env_name:
        return (np.abs(np.linalg.norm(state[-3:])) > .02)
    return (np.abs(state[1]) <= .2)


def set_state(eval_env, state):
    if 'Reacher' in eval_env.unwrapped.spec.id:
        adjust = (state[0:2] < 0) * np.pi 
        eval_env.set_state(np.concatenate([np.arctan(state[2:4]/state[0:2]) + adjust, eval_env.get_body_com("target")[:2]]), np.concatenate([state[6:8], np.array([0,0])]))
        state[4:6] = eval_env.get_body_com("target")[:2] # target position
        state[-3:] = eval_env.get_body_com("fingertip") - eval_env.get_body_com("target")  # fingertip - target
    else:
        eval_env.set_state(state[0:2], state[2:4])
    return state


def visualize(policy, env_name):
  eval_env = gym.make(env_name)
  state = eval_env.reset()
  total_reward = 0
  for i in range(200):
      done = not (np.isfinite(state).all() and done_condition(env_name, state))

      if done:
          print(f"Reward {total_reward}")
          return total_reward
      else:
          total_reward += 1 

      print(i)
      img = cv2.cvtColor(eval_env.render("rgb_array"), cv2.COLOR_BGR2RGB)
    #  cv2.imwrite(f"videos/imitation/{i}.png", img)
      cv2.imshow("Model prediction", img)
      cv2.waitKey(1)

      state = np.squeeze(policy.select_goal(state))
      try:
        state = set_state(eval_env, state)
        time.sleep(.01)
      except:
        pass

  return total_reward


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                    if 'Fetch' in env_name:
                        state = state['observation']

                    action = policy.select_action(np.array(state))
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward



def eval_model(policy, env_name, seed, eval_episodes=10):
        return visualize(policy, env_name)

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
        parser.add_argument("--save_dir", default="bco_expert")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
        parser.add_argument("--randomness", default=0, type=float)                # Randomness of expert
        parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
        parser.add_argument("--tau", default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--visualize", action="store_true")        # Visualize model predictions
        parser.add_argument("--expert", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
        args = parser.parse_args()

        file_name = f"{args.policy}_{args.env}_{args.randomness}_{args.expert}_{args.seed}"
        demo_name = f"{args.policy}_{args.env}_{args.randomness}_True_0"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Randomness: {args.randomness}, Expert: {args.expert}, Seed: {args.seed}")
        print("---------------------------------------")

        results_dir = os.path.join(args.save_dir, "bco_results")
        models_dir = os.path.join(args.save_dir, "models")

        if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        if args.save_model and not os.path.exists(models_dir):
                os.makedirs(models_dir)

        results_file = open(os.path.join(results_dir, file_name + ".txt"), "w")
        env = gym.make(args.env)

        # Set seeds
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
      
        if "Fetch" in args.env:
            state_dim = env.observation_space['observation'].shape[0]
        else:
            state_dim = env.observation_space.shape[0]

        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])

        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args.discount,
                "tau": args.tau,
        }

        # Initialize policy
        if args.policy == "D3G":
                kwargs["policy_freq"] = args.policy_freq

        # Load policy
        ground_truth_policy = TD3.TD3(**kwargs)
        TD3_file_name = "TD3_" + args.env + "_0"

        ground_truth_policy.load(f"./models/{TD3_file_name}")

        done = False 
        total_reward = 0 

        # Collect expert observations
        expert_buffer = utils.GoalReplayBuffer(state_dim, action_dim)
        expert_obs_buffer = utils.ObservationalBuffer(state_dim, action_dim)
        state = env.reset() 
        avg_reward = 0
        expert_episodes = 0
        evaluations = []

        if os.path.exists(f"bco_expert/{demo_name}_expert_buffer.pkl"):
            expert_buffer = pickle.load(open(f"bco_expert/{demo_name}_expert_buffer.pkl", "rb"))
            expert_obs_buffer = pickle.load(open(f"bco_expert/{demo_name}_expert_obs_buffer.pkl", "rb"))
        else:
          for t in range(0, 1000000):
              if np.random.uniform(0,1) < args.randomness:
                  action = env.action_space.sample()
              else:
                  action = ground_truth_policy.select_action(state)

              next_state, reward, done, _ = env.step(action) 
              expert_buffer.add(state, action,  next_state, next_state, reward, 0, done)
              expert_obs_buffer.add(state, next_state) 

              state = next_state
              total_reward += reward 

              if done: 
                  print(f"{t}: {total_reward}")
                  avg_reward += total_reward
                  state, done = env.reset(), False
                  total_reward = 0
                  expert_episodes += 1.

          print(f"Average reward {avg_reward / expert_episodes}")
          expert_file = open(os.path.join(results_dir, file_name + "_expert_file.txt"), "w")
          expert_file.write(f"{expert_episodes} {avg_reward / expert_episodes}\n")
          expert_file.close()
          pickle.dump(expert_buffer, open(f"bco_expert/{demo_name}_expert_buffer.pkl", "wb"))
          pickle.dump(expert_obs_buffer, open(f"bco_expert/{demo_name}_expert_obs_buffer.pkl", "wb"))

        # Train QSS and collect observations
        if not args.expert:
            kwargs["summary_name"] = args.env 
            kwargs["batch_size"] = args.batch_size
            d3g_policy = ObservationalD3G.D3G(**kwargs)
            for t in range(1000000):
                    # Store data in replay buffer
                    d3g_policy.train(expert_buffer)

                    if t % 1000 == 0:
                        print(f"QSS training: {t}")

                        if args.visualize:
                            eval_model(d3g_policy, args.env, args.seed)
      
        if args.expert:
          bco_policy = bco.BCO(state_dim=state_dim, action_dim=action_dim, max_action=max_action, batch_size=args.batch_size)
        else:
          bco_policy = d3g_policy

        bco_buffer = utils.ReplayBuffer(state_dim, action_dim)

        for t in range(100):
            print(f"BCO t {t}")
            # Evaluate policy
            evaluation = eval_policy(bco_policy, args.env, args.seed)
            evaluations.append(evaluation)
            np.save(f"{results_dir}/{file_name}", evaluations)
            results_file.write(f"{t} {evaluation}\n")
            results_file.flush()

            # 1) Take steps in environment
            state = env.reset()

            for _ in range(1000):
                action = (
                        bco_policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

                next_state, reward, done, _ = env.step(action)
                bco_buffer.add(state, action, next_state, reward, done)

                state = next_state 

                if done:
                    state = env.reset()

            # 2) Train inverse dynamics with experience from (1)
            for actor_t in range(10000):
                bco_policy.train_actor(bco_buffer) 

            # 3) Run BC on observations
            if args.expert:
              for bc_t in range(10000):
                bco_policy.train_bc(expert_obs_buffer)

        results_file.close() 
