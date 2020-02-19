from collections import defaultdict
import numpy as np 
import random 
import cv2 

import argparse
import pickle 
import os 

if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("stored_models"):
    os.makedirs("stored_models")

parser = argparse.ArgumentParser()
parser.add_argument("--shuffled", action="store_true")       

args = parser.parse_args()

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

empty_set = set() 

if args.shuffled:
    UP = 3
    DOWN = 2
    LEFT = 1
    RIGHT = 0
else:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def update_I(state, action, next_state):
    ID[str(state), str(next_state)].add(action)

def I(state, next_state):
    actions = ID[str(state), str(next_state)]
    if actions == empty_set:
        return np.random.randint(num_actions)
    else:
        return random.choice([action for action in actions])


def N(state):
    row = state[0]
    col = state[1] 

    neighbors = []
    new_row = max(0, row - 1) 
    neighbors.append([new_row, col])

    new_row = min(SIZE, row + 1)
    neighbors.append([new_row, col])

    new_col =  max(0, col - 1)
    neighbors.append([row, new_col])

    new_col = min(SIZE, col + 1)
    neighbors.append([row, new_col])

    return neighbors 


def step(state, action, steps):
    row = state[0]
    col = state[1] 

    new_row = row
    new_col = col

    if action in range(UP, num_actions, 4):
        new_row = max(0, row - 1) 
    elif action in range(DOWN, num_actions, 4):
        new_row = min(SIZE, row + 1)
    elif action in range(LEFT, num_actions, 4):
        new_col =  max(0, col - 1)
    elif action in range(RIGHT, num_actions, 4):
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
        reward = -1

    if steps == MAX_STEPS - 1:
        max_reached = 1

    return next_state, reward, terminal, max_reached


def policy(state):
    neighbors = N(state)
    best_state_index = np.argmax([Q[str(state), str(neighbor)] for neighbor in neighbors])
    best_state = neighbors[best_state_index]

    return I(state, best_state)

def eval_policy():
    state = [0, 0] 
    steps = 0
    total_reward = 0
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

    return total_reward


def update(state, action, reward, next_state, terminal):
    next_q_values = [Q[str(next_state), str(neighbor)] for neighbor in N(next_state)]

    target = reward + GAMMA * max(next_q_values) * (1 - terminal)
    s = str(state)
    next_s = str(next_state)

    Q[(s, next_s)] = Q[(s, next_s)] + ALPHA * (target - Q[(s, next_s)])
    update_I(state, action, next_state)

def train():
    for trial in range(50):
        np.random.seed(trial)
        random.seed(trial)

        global Q
        global ID 

        ID = defaultdict(lambda: set()) 
        if args.shuffled:
            vanilla_file = open("results/model_shuffle_" + str(num_actions) + "_"  + str(trial) + ".txt", "w")
            with open(f'stored_models/qss/{trial}.txt', 'r') as f:
                Q = eval(f.read())
        else:
            vanilla_file = open("results/model_original_" + str(num_actions) + "_"  + str(trial) + ".txt", "w") 
            Q = defaultdict(lambda: .001)


        epsilon = INITIAL_EPSILON

        for episode in range(EPISODES + 1):
            terminal = 0
            max_steps = 0
            state = [0, 0] 
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
                update(state, action, reward, next_state, terminal) 

                state = next_state
                steps += 1
                total_reward += reward 

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if not args.shuffled:
            with open(f'stored_models/qss/{trial}.txt', 'w+') as f:
                f.write(str(dict(Q)))

train()
