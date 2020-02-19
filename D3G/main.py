""" Modifications Copyright (c) 2019 Uber Technologies, Inc. """

import numpy as np
import cv2
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import D3G
import Standard_QSS

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

      cv2.imshow("Model prediction", img)
      cv2.waitKey(1)

      state = np.squeeze(policy.select_goal(state))
      try:
        state = set_state(eval_env, state)
        time.sleep(.01)
      except:
        pass

  return total_reward


def make_env(env_name):
    return gym.make(env_name)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
        eval_env = make_env(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                    action = policy.select_action(np.array(state))
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward


if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
        parser.add_argument("--save_dir", default=".")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
        parser.add_argument("--train_vae", default=1e4, type=int)       # Time steps for training vae
        parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
        parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)                 # Discount factor
        parser.add_argument("--tau", default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--visualize", action="store_true")        # Visualize model predictions
        parser.add_argument("--is_discrete", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
        args = parser.parse_args()

        if args.load_model:
          file_name = f"{args.policy}_{args.env}_{args.seed}"
        else:
          file_name = f"{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")


        results_dir = os.path.join(args.save_dir, "results")
        models_dir = os.path.join(args.save_dir, "models")

        if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        if args.save_model and not os.path.exists(models_dir):
                os.makedirs(models_dir)

        env = make_env(args.env)

        # Set seeds
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]

        if args.is_discrete:
            action_dim = env.action_space.n
            max_action = float(action_dim)
        else:
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args.discount,
                "is_discrete": args.is_discrete,
                "tau": args.tau,
        }

        # Initialize policy
        if args.policy == "TD3":
                # Target policy smoothing is scaled wrt the action scale
                kwargs["policy_noise"] = args.policy_noise * max_action
                kwargs["noise_clip"] = args.noise_clip * max_action
                kwargs["policy_freq"] = args.policy_freq
                policy = TD3.TD3(**kwargs)
        elif args.policy == "OurDDPG":
                policy = OurDDPG.DDPG(**kwargs)
        elif args.policy == "D3G":
                kwargs["policy_freq"] = args.policy_freq
                policy = D3G.D3G(**kwargs)
        elif args.policy == "Standard_QSS":
                kwargs["policy_freq"] = args.policy_freq
                policy = Standard_QSS.Standard_QSS(**kwargs)

        if args.load_model != "":
                policy_file = file_name if args.load_model == "default" else args.load_model
                policy.load(f"{models_dir}/{policy_file}")

        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.is_discrete)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy, args.env, args.seed)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(args.max_timesteps)):
                episode_timesteps += 1

                # Select action randomly or according to policy
                if t < args.start_timesteps:
                        action = env.action_space.sample()
                elif args.is_discrete:
                        if np.random.uniform(0,1) < .1:
                            action = env.action_space.sample()
                        else:
                            action = policy.select_action(np.array(state))
                else:
                        action = (
                                policy.select_action(np.array(state))
                                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)

                # Perform action
                next_state, reward, done, _ = env.step(action)
                done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

                if t >= args.start_timesteps:
                        policy.train(replay_buffer, args.batch_size)

                if done:
                        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                        # Reset environment
                        state, done = env.reset(), False
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num += 1

                # Evaluate episode
                if (t + 1) % args.eval_freq == 0:
                        evaluation = eval_policy(policy, args.env, args.seed)
                        evaluations.append(evaluation)

                        np.save(f"{results_dir}/{file_name}", evaluations)
                        
                        if args.visualize:
                            visualize(policy, args.env)

                        elif args.save_model: policy.save(f"{models_dir}/{file_name}")
