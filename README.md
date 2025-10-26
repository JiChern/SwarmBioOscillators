# Graph-CPG
A framework that can generate synchronous waveforms with arbitrary phase-lags in arbitrary size of coupled-oscillator-system. Training is very simple!


[Ji Chen](mailto:ji.chenuk@gmail.com), Song Chen, Yunhan He,  [Li Fan](mailto:fanli77@zju.edu.cn) and [Chao Xu](mailto:cxu@edu.zju.cn)


## Introduction
This repository includes code implementations of the paper titled "Learning Emergent Synchronization in Coupled Oscillators via Graph Attention and Reinforcement Learning" .

## Coupled oscillators controlled by graph attention mechanism
We reconceptualize the problem of waveform generation in coupled-oscillator systems from the viewpoint of swarm intelligence. Our objective is to enable each unit within the coupled system to learn what it should attend to in order to achieve collective objectives. This approach aligns closely with contemporary research on graph attention mechanisms. Based on our concept, each unit learns a distributed strategy where the input is the decomposition of the global goal from the unit's local perspective, and the output is the attention it allocates to other units. 

Here, we propose the graph-CPG model, a concrete implementation of our macro-level concept within a two-dimensional coupled oscillator system. Our task is to enable a coupled oscillator system to generate corresponding oscillatory modes based on user-specified desired phase inputs. For a system with $N$ units, we define the desired phase vector $x_{\text{dp}} = \{\theta_i\} \in [0, 2\pi]^N$, where $\theta_i$ represents the desired phase lag between the $i$-th node and the first node. For each node $i$, we define its attention to a neighbor $j$  as $\alpha_{i,j}$. The system of coupled 2D oscillators is then governed by: 

```math
	\dot{\mathbf{x}}_i = f(\mathbf{x}_i) + \text{clamp}\left[\text{MLP}\left(\frac{1}{K}\sum_{k}\sum_{j \in \mathcal{N}(i)} \alpha_{i,j}^k \Theta_t \mathbf{x}_j\right), -1, 1\right],  
```
where $\mathbf{x}_i \in \mathbb{R}^2$ denotes the state vector of the $i$-th node, and $f: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ models the internal dynamics (e.g., Hopf or Van der Pol oscillator dynamics). The structure inside the multilayer perceptron (MLP) implements a graph attention message-passing mechanism \cite{velivckovic2017graph, brody2021attentive}. To bridge the goal and attention generation, the attention of the $i$-th node to the $j$-th node (for the $k$-th attention head) is computed via a shared attention function $a: \mathbb{R}^{F} \times \mathbb{R}^{F} \rightarrow \mathbb{R}$:  

```math
	\alpha_{i,j}^k = a\left(\Theta_s^{\text{dp},k} \mathbf{x}_{\text{dp}} + \Theta_s^k \mathbf{x}_i, \Theta_t^{\text{dp},k} \mathbf{x}_{\text{dp}} + \Theta_t^k \mathbf{x}_j\right),  
``` 

where $\Theta_s^{\text{dp},k}$, $\Theta_t^{\text{dp},k}$, $\Theta_s^k$, and $\Theta_t^k$ are learnable projection matrices. These matrices transform node states $x_i, x_j$ and the desired phase relationships $x_{\text{dp}}$ into a shared $F$-dimensional feature space. The additive structure incorporates a positional encoding technique inspired by Transformers, allowing the attention coefficients to adapt dynamically to the system's states.

# Installation
Prerequisites: Ubuntu 20.04, Miniforge toolkits, torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1, torch-scatter torch-sparse, torch-cluster, torch-spline-conv, pyg-lib and torch-geometric.

## Step 1: Install Miniforge (or Mambaforge) for Conda/Mamba:
Download and install Miniforge (a minimal Conda installer) from https://github.com/conda-forge/miniforge. Choose the version for your OS (e.g., Linux, macOS, Windows).
Follow the installer prompts. This sets up conda and mamba (a faster alternative to conda).
Verify: Run mamba --version in your terminal.

## Step 2: Source and initialize the Conda and Mamba Scripts
```console
source ~/miniforge3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/mamba.sh
~/miniforge3/bin/conda init
```
Restart terminal
```console
mamba --version
```

## Step3: Setup all packages in virtual environment

### Create virtual environment
```console
mamba create -n g_cpg
mamba activate g_cpg
conda config --env --add channels conda-forge
conda config --env --remove channels defaults
conda config --env --add channels robostack-noetic
```

### Install ros and pybullet for robot experiments 
```console
conda install ros-noetic-desktop
conda deactivate
conda activate g_cpg
conda install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
pip install pybullet
```

### Install dependencies for graph-CPG framework
```console
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-geometric
```

### Unzip the repo in your local directory.

# Train the model
Use train.py, you can train graph-CPG with differnent number of attention heads and dimensionalities of feature space.
```console
conda activate g_cpg
python train.py --heads=8 --fd=64
```
# Test the model
```console
conda activate g_cpg
python test_model.py
```
# Simulate a centipede robot in pybullet with graph-CPG
<p align="center">
  <img src="https://github.com/JiChern/Graph-CPG/blob/main/fig/centipede.gif?raw=true" alt="Sublime's custom image"/>
</p>


## Step 1: execute the gait generator
Open a terminal
```console
conda activate g_cpg
cd simulation
python gait_generator.py --cell_num=20  #you can adjust --cell_num to any odd number <= 34 (if more than 34 legs, pybullet cannot hanle these much joints by default settings)
```
## Step 2: run the simulation script
Open another terminal
```console
conda activate g_cpg
cd simulation
python sim_robot.py --seg_num=10  #you can adjust --cell_num to any number <= 17 (if more than 34 legs, pybullet cannot hanle these much joints by default settings)
```
## Step 3: Adjust the turning of the centipede
After the robot is moving, you can adjust its turning within [-1,1]. Open another terminal:
```console
conda activate g_cpg
rostopic pub /turning std_msgs/Float32 "data: 0.8" 
```

## References

<a id="1">[1]</a> 
Haitao Yu and Haibo Gao and Liang Ding and Mantian Li and Zongquan Deng and Guangjun Liu (2016). 
Gait generation with smooth transition using CPG-based locomotion control for hexapod walking robot. 
IEEE Transactions on Industrial Electronics, 63, 5488-5500.

<a id="1">[2]</a> 
Ludovic Righetti and Auke Jan Ijspeert (2008). 
Pattern generators with sensory feedback for the control of quadruped locomotion. 
2008 IEEE International Conference on Robotics and Automation, 819-824.

<a id="1">[3]</a> 
Wei Xiao and Wei Wang (2014). 
Hopf oscillator-based gait transition for a quadruped robot. 
2014 IEEE International Conference on Robotics and Biomimetics, 2074-2079.


