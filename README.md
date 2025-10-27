# Swarm-Inspired Emergent Synchronization in Biologically Coupled Dynamical Systems
A framework that can generate synchronous waveforms with arbitrary phase-lags in arbitrary size of coupled-oscillator-system. Training is very simple!


[Ji Chen](mailto:ji.chenuk@gmail.com), [Song Chen](mailto:math.cs.zju@gmail.com), [Chengzhang Gone](mailto:12532009@zju.edu.cn),  [Li Fan](mailto:fanli77@zju.edu.cn) and [Chao Xu](mailto:cxu@edu.zju.cn)


## Introduction
This repository includes code implementations of the paper titled "Swarm-Inspired Emergent Synchronization in Biologically Coupled Dynamical Systems" .

## Coupled oscillators controlled by distrubuted attention
In natural swarms---flocking birds, schooling fish, insect colonies---global coordination emerges from local perception and decision-making, without centralized control. Individual agents selectively attend to neighbors based on context, enabling adaptive and scalable collective behavior. Inspired by this, we treat each oscillatory units in a CPG as an intelligent agent that learns how to interact: rather than analyzing the coupling functions, we endow units with attention mechanisms that dynamically weight their influence on neighbors, guided by a population-level objective. This transforms the classical problem of analyzing CDSs from a macrostructural perspective to a microstructural one. In doing so, rigid synchronization becomes an emergent, adaptive process---mirroring biological CPGs while unlocking computational scalability.

This principle underpins graph-CPG, a coupled oscillator framework where interactions are governed by learned attention (Fig.~\ref{fig:intro}d). Analogous to how flocking emerges from simple neighbor-following rules, graph-CPG produces versatile global synchronous behaviors through local attention-weighted coupling. In this model, each oscillator $i$ evolves according to intrinsic dynamics $f(\cdot)$—such as Hopf or Van der Pol oscillators—and an external coupling term $\mathbf{a}_i$ encoding adaptive interactions:

```math
	\dot{\mathbf{x}}_i = f(\mathbf{x}_i) + \text{clamp}\left[\text{MLP}\left(\frac{1}{K}\sum_{k}\sum_{j \in \mathcal{N}(i)} \alpha_{i,j}^k \Theta_t \mathbf{x}_j\right), -1, 1\right],  
```
for $i=1,2,\dots,N$, where the dense layer and clamping operation regularize the external coupling term for numerical stability. Concrete definitions of the parameters in Equation~\eqref{eq:graph-cpg-main} and the details of graph-CPG can be found in Methods. The attention coefficients $\alpha_{i,j}^k$ weight the importance of the neighbor node $j$ to the source node $i$, and depend on the current states of both nodes $(\mathbf{x}_i, \mathbf{x}_j)$ as well as their desired phase lags to achieve synchronization $(\theta_i, \theta_j)$. These desired phase lags for achieving synchronization are collected in a vector $\mathbf{x}_{\text{dp}} = [\theta_1, \dots, \theta_N]$, which defines the target emergent phase behavior of the oscillator network, with each oscillator maintaining a specified phase lag $\theta_i$ relative to the first oscillator.

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
  <img src="https://github.com/JiChern/SwarmBioOscillators/blob/main/fig/centipede.gif?raw=true" alt="Sublime's custom image"/>
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

<!-- <a id="1">[1]</a> 
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
2014 IEEE International Conference on Robotics and Biomimetics, 2074-2079. -->


