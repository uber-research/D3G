from collections import defaultdict
import numpy as np 
import random 
import cv2 
import os 

if not os.path.exists("results"):
    os.makedirs("results")

num_actions = 4
SIZE = 10
GAMMA = .99 
ALPHA = .01

EXPLORE = 100000
FINAL_EPSILON = .1 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon

EPISODES = 50000
MAX_STEPS = 500

DISPLAY = False 
actions = range(num_actions)


def update(state, action, reward, next_state, terminal):
    next_q_values = []

    for next_action in range(num_actions):
        next_q_values.append(Q[(next_state, next_action)])

    target = reward + GAMMA * max(next_q_values) * (1 - terminal)
    Q[(state, action)] = Q[(state, action)] + ALPHA * (target - Q[(state, action)])
   

def step(state, action, steps):
    row = state[0]
    col = state[1] 

    new_row = row
    new_col = col

    if action in range(0, num_actions, 4):
        new_row = max(0, row - 1) 
    elif action in range(1, num_actions, 4):
        new_row = min(SIZE, row + 1)
    elif action in range(2, num_actions, 4):
        new_col =  max(0, col - 1)
    elif action in range(3, num_actions, 4):
        new_col = min(SIZE, col + 1)
    else:
        print("Wrong action")
        exit() 

    next_state = [new_row, new_col] 
    max_reached = 0

    if next_state == [SIZE, SIZE]:
        reward = 1
        terminal = 1
    else:
        terminal = 0
        reward = 0

    if steps == MAX_STEPS - 1:
        max_reached = 1

    # Add wind
    if row == SIZE and (action == 1 or np.random.uniform(0,1) < .5):
        next_state = [row + 1, col]
        terminal = 1
        reward = 0

    return next_state, reward, terminal, max_reached


def policy(state):
    return np.argmax([Q[(str(state), action)] for action in actions])


def eval_policy():
    total_reward = 0

    for _ in range(10):
        state = [SIZE, 0] 
        steps = 0
        terminal = 0
        max_steps = 0

        while not terminal and not max_steps:
            if DISPLAY:
                row, col = state
                img = np.zeros([SIZE + 1, SIZE + 1, 3])
                img[row, col] = [0, 255, 0]
                img[SIZE, SIZE] = [255, 0, 0]

                cv2.imshow("", img)
                cv2.waitKey(1)

            action = policy(state)
            next_state, reward, terminal, max_steps = step(state, action, steps)

            steps += 1
            total_reward += reward

            state = next_state 

    return total_reward / 10.


def train():
    for trial in range(10):
        np.random.seed(trial)
        random.seed(trial)

        vanilla_file = open("results/vanilla_" + str(num_actions) + "_"  + str(trial) + ".txt", "w") 
        global Q
        Q = defaultdict(lambda: .001) 
        epsilon = INITIAL_EPSILON

        for episode in range(EPISODES + 1):
            terminal = 0
            max_steps = 0
            state = [SIZE, 0] 
            steps = 0 
            total_reward = eval_policy() 

            print(str(episode) + " " + str(total_reward) + " " + str(epsilon))
            vanilla_file.write(str(episode) + " " + str(total_reward) + "\n")

            while not terminal and not max_steps:
                if np.random.uniform(0,1) < epsilon:
                    action = random.choice(actions)
                else:
                    action = policy(state) 

                next_state, reward, terminal, max_steps = step(state, action, steps) 
                update(str(state), action, reward, str(next_state), terminal) 

                state = next_state
                steps += 1
                total_reward += reward 

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

train()
