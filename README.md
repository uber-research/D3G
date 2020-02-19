# Estimating Q(s,s') with Deep Deterministic Dynamics Gradients

<p align=center>
<img src="https://github.com/uber-research/D3G/blob/master/resources/trajectory.gif" width="250">&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/uber-research/D3G/blob/master/resources/learned_pendulum.gif" width="250">&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/uber-research/D3G/blob/master/resources/learned_reacher.gif" width="250">
</p>

<div align="center">
  Figure 1: Model predictions learned by D3G.
</div>
</br>
</br>

Official PyTorch implementation of Deep Deterministic Dynamics Gradients. Code is based heavily on the Twin Delayed DDPG [implementation](https://github.com/sfujim/TD3). For research purpose only. Support and/or new releases may be limited.

## Running toy QSS problems
The stochastic action results can be reproduced by running:
```
cd toy_problems/stochastic_actions && python model_gridworld.py --stochasticity {rand}
```
Where rand is either 0, .25, .5, or .75.

The redundant action results can be reproduced by running:
```
cd toy_problems/redundant_actions && python model_gridworld.py
```

The shuffled action results can be reproduced by running:
```
cd toy_problems/shuffled_actions && python model_gridworld.py && python model_gridworld.py --shuffled
```

## Running D3G
The paper results can be reproduced by running:
```
cd D3G && ./mujoco_experiments/run_D3G_experiments.sh
```

The model predictions for Reacher-v2 and InvertedPendulum-v2 can be visualized. To see this, run:
```
cd D3G && python main.py --policy D3G --env Reacher-v2 --visualize
```

## Running Learning from Observation
The paper results can be reproduced by running:
```
cd D3G && ./lfo_experiments/run_D3G_experiments
```

The model predictions for Reacher-v2 and InvertedPendulum-v2 can be visualized here too. To see this, run:
```
cd D3G && python learn_from_observation.py --policy D3G --env Reacher-v2 --visualize
```
