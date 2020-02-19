from collections import defaultdict
import numpy as np 
import random 
import cv2 

import argparse
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument("--stochasticity", type=float)       

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

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3



def I(state, next_state):
    row = state[0]
    col = state[1]

    new_row = next_state[0]
    new_col = next_state[1]

    if new_row < row:
        return UP
    elif new_row > row:
        return DOWN
    elif new_col < col:
        return LEFT
    elif new_col > col:
        return RIGHT
    elif state == next_state:
        if row == 0 and col == 0:
            return random.choice([UP, LEFT]) 
        elif row == SIZE and col == 0:
            return random.choice([DOWN, LEFT])
        elif row == 0 and col == SIZE:
            return random.choice([UP, RIGHT])
        elif row == SIZE and col == SIZE:
            return random.choice([RIGHT, DOWN])
        elif row == 0:
            return UP
        elif row == SIZE:
            return DOWN 
        elif col == 0:
            return LEFT
        elif col == SIZE:
            return RIGHT 


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

    if np.random.uniform(0,1) < args.stochasticity:
        stochastic_action = np.random.randint(num_actions)
    else:
        stochastic_action = action

    if stochastic_action in range(UP, num_actions, 4):
        new_row = max(0, row - 1) 
    elif stochastic_action in range(DOWN, num_actions, 4):
        new_row = min(SIZE, row + 1)
    elif stochastic_action in range(LEFT, num_actions, 4):
        new_col =  max(0, col - 1)
    elif stochastic_action in range(RIGHT, num_actions, 4):
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
    total_reward = 0

    for _ in range(10):
        state = [0, 0] 
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

def update(state, action, reward, next_state, terminal):
    next_q_values = [Q[str(next_state), str(neighbor)] for neighbor in N(next_state)]

    target = reward + GAMMA * max(next_q_values) * (1 - terminal)
    s = str(state)
    next_s = str(next_state)

    Q[(s, next_s)] = Q[(s, next_s)] + ALPHA * (target - Q[(s, next_s)])

def train():
    for trial in range(10):
        np.random.seed(trial)
        random.seed(trial)

        global Q

        vanilla_file = open("results/model_stochastic_" + str(args.stochasticity) + "_" + str(num_actions) + "_"  + str(trial) + ".txt", "w") 
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

        with open(f'stored_models/qss/{trial}_{args.stochasticity}.txt', 'w+') as f:
            f.write(str(dict(Q)))

train()
