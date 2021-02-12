# Exploration of Hindsight Experience Replay with TD3

### [Read our blog post!](https://aimless-agents.github.io/articles/2021-02/HER)

In this project we set out to reproduce results from Andrychowicz et al.'s [Hindsight Experience Replay paper](https://arxiv.org/abs/1707.01495), using Fujimoto et al.'s original [TD3 implementation](https://github.com/sfujim/TD3) as a baseline. Original TD3 paper linked [here](https://arxiv.org/abs/1802.09477).

We tested our implementation on OpenAI's [FetchReach](https://gym.openai.com/envs/FetchReach-v0/) environment, and variations of MuJoCo's [Reacher](https://gym.openai.com/envs/Reacher-v2/) environment. 

### Usage

#### Training
To train a model on a certain environment, run:
```
python main.py --env <gym environment name> 
```
To save the model during training, also include the `--save_model` flag. Additional flags are detailed in `parser.py`.

#### Plotting
 
To plot the agent's learning curve, run:
```
python plot_script.py <same args used to run main.py>
```

#### Visualizing

To visualize the agent in action, run:
```
python record_policy_video.py <same args used to run main.py>
```
or
```
python record_policy_video.py --f <name of saved actor file>
```
This requires the model to have been saved.

### Results

Results and analyses can be found in [our blog post](https://aimless-agents.github.io/articles/2021-02/HER).

