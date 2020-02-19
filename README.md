# Estimating Q(s,s') with Deep Deterministic Dynamics Gradients

PyTorch implementation of Deep Deterministic Dynamics Gradients. For research purpose only. Support and/or new releases may be limited.

Our code is based heavily on the Twin Delayed DDPG [implementation](https://github.com/sfujim/TD3).

### Running toy QSS problems
The stochastic action results can be reproduced by running:
```
cd toy_problems/stochastic_actions && python model_gridworld.py --randomness rand
```
Where rand is either 0, .25, .5, or .75.

The redundant action results can be reproduced by running:
```
cd toy_problems/redundant_actions && python model_gridworld.py
```

The shuffled action results can be reproduced by running:
```
cd toy_problems/shuffled_actions && python model_gridworld.py
```

### Running D3G
The paper results can be reproduced by running:
```
cd D3G && ./mujoco_experiments/run_D3G_experiments.sh
```

### Running Learning from Observation
The paper results can be reproduced by running:
```
cd D3G && ./lfo_experiments/run_D3G_experiments
```
