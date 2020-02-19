# Estimating Q(s,s') with Deep Deterministic Dynamics Gradients

PyTorch implementation of Deep Deterministic Dynamics Gradients.

Our code is based heavily on the Twin Delayed DDPG [implementation](https://github.com/sfujim/TD3).

### Running toy QSS problems
The stochastic action results can be reproduced by running:
```
cd toy_problems/stochastic_actions && python model_gridworld.py
```

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
cd TD3 && ./mujoco_experiments/run_D3G_experiments.sh
```

### Running Learning from Observation
The paper results can be reproduced by running:
```
cd TD3 && ./lfo_experiments/run_D3G_experiments
```

For research purpose only. Support and/or new releases may be limited.
