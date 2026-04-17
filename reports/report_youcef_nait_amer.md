# Individual Report — Youcef NAIT AMER

**Project:** Reinforcement Learning on highway-v0
**Course:** Mention IA — CentraleSupelec
**Instructor:** Hédi Hadiji

---

## 1. Personal Contribution

My contribution focused primarily on building the models, training and evaluating them on selected seeds, and then presenting and interpreting the results for the oral defense.

### 1.1 Getting Started with the Project

As a first step, each team member independently attempted to implement a simple DQN, drawing on what we had covered in class and applying it to the project. We also had to familiarize ourselves with the highway-v0 environment, its action space, configurable parameters, and overall dynamics in order to develop an informed understanding before moving to more complex agents.

### 1.2 Environment

The environment consists of a highway populated with vehicles to avoid. The agent controls a car whose objective is to travel as fast as possible without colliding. At each timestep, five discrete actions are available: accelerate, decelerate, turn right, turn left, and maintain lane.

The reward function encourages high speed, penalizes collisions heavily, and applies a mild penalty for lane changes.

### 1.3 Model Architectures

**Primary CNN (used as the Q-function):**

The network is structured as follows:

- **Input layer:** kinematic observations reshaped into a 2D grid (5 vehicles × 5 features)
- **Conv2D:** 16 filters, kernel size 3, ReLU activation
- **Conv2D:** 32 filters, kernel size 3, ReLU activation
- **Flatten + fully-connected layers:** sizes 256 and 128, ReLU activation
- **Output layer:** 5 discrete actions (DiscreteMetaAction)

The choice of a convolutional architecture is motivated by the tabular structure of the kinematic observation: each row represents one vehicle, and convolutions can capture local patterns between adjacent vehicles in this representation.

**Deeper CNN (ablation):**

We also implemented a deeper variant with three convolutional layers to compare against the primary CNN:

- **Input layer:** kinematic observations reshaped into a 2D grid (5 vehicles × 5 features)
- **Conv2D:** 16 filters, kernel size 3, Batch Normalization, ReLU activation
- **Conv2D:** 32 filters, kernel size 3, Batch Normalization, ReLU activation
- **Conv2D:** 32 filters, kernel size 3, Batch Normalization, ReLU activation
- **Flatten + fully-connected layers:** sizes 256 and 128, ReLU activation
- **Output layer:** 5 discrete actions (DiscreteMetaAction)

**SB3 MlpPolicy:**

- **Input:** FlattenExtractor (50-dimensional vector)
- **Linear layer:** output size 64, ReLU activation
- **Linear layer:** output size 64, ReLU activation
- **Output layer:** 5 discrete actions

### 1.4 Double DQN

The Double DQN modification addresses the overestimation bias inherent in standard Q-learning. Rather than using the target network to both select and evaluate the greedy action, the online network selects the best action and the target network evaluates it:
y = r + γ · Q_target(s', argmax_a Q_online(s', a))
This change is minimal at the code level, a single line of logic in the target computation, but produces a measurable gain in training stability.

---

## 2. Experiments

Training runs were conducted independently by each team member. On my end, I trained all four model variants (standard DQN, DDQN, deeper CNN DQN, and SB3) on seed 44 for 500 episodes. I subsequently re-ran the DQN and DDQN training on the same seed for 2 000 episodes to assess longer-horizon behavior.

All models were evaluated over 50 episodes following training.

---

## 3. Results and Analysis

After evaluation, I rendered each trained model as a GIF of several episodes to obtain a qualitative view of agent behavior alongside the quantitative metrics.

The reward curves are notably noisy across all models, though they exhibit an overall upward trend. Of all variants tested, the standard DQN with the simpler primary CNN performed best on seed 44. The deeper CNN did not yield a consistent improvement, suggesting that additional convolutional depth does not translate into better performance under the current training budget and observation structure nor did the DDQN which probably didn't converge.

The GIF visualizations also revealed mild instabilities in the learned policies: agents occasionally commit to suboptimal lane changes or fail to adjust speed in time, consistent with the reward variance observed during training.

---

## 4. Limitations

- Training runs were limited to 500 episodes initially, which may be insufficient for stable convergence given the noisy reward signal in highway-v0. The extended 2 000-episode runs provided more stable policies but remain below what the environment likely requires for fully robust behavior.
- The deeper CNN was tested only on seed 44. Broader evaluation across multiple seeds would be necessary to draw conclusions about whether batch normalization and additional depth offer a reliable benefit.
- The comparison between the CNN-based custom agents and the SB3 MlpPolicy conflates two factors: the training framework and the network architecture. Isolating these effects would require either implementing a CNN policy within SB3 or using an MLP for the custom agent.
