# Swarm-inspired Emergent Synchronizer (SIES)
This is a study on graph dynamical systems, covering two fundamental aspects:


(1) It introduces a learning-based coupled dynamical system (SIES) that models local coupling quantities as signed graph attention. This enables the dynamical system to emerge synchronization modes with arbitrary phase differences across networks of any scale. Experiments demonstrate that SIES can function as a central pattern generator (CPG) to produce gaits for multi-legged robots with any number of legs, while also enhancing the robot’s resiliency to limb damage.


(2) Leveraging SIES’s rich oscillatory modes and high-quality phase dynamics in networks of any size, it is directly applied as a graph neural network for node classification tasks. It achieves SOTA performance on heterophilous graph datasets. Notably, SIES-GNN can act like a “centrifuge,” continuously rotating the states of all nodes and segregating nodes of different classes into distinct spatial regions with different phase differences.


[Ji Chen](mailto:ji.chenuk@gmail.com), [Song Chen](mailto:math.cs.zju@gmail.com), [Chengzhang Gong](mailto:12532009@zju.edu.cn),  [Li Fan](mailto:fanli77@zju.edu.cn) and [Chao Xu](mailto:cxu@edu.zju.cn)



# Installation
Prerequisites: 10GB free space, Ubuntu 20.04, Miniforge toolkits, torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1, torch-scatter torch-sparse, torch-cluster, torch-spline-conv, pyg-lib and torch-geometric.

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
mamba create -n sies
mamba activate sies
conda config --env --add channels conda-forge
conda config --env --remove channels defaults
conda config --env --add channels robostack-noetic
```

### Install ros and pybullet for robot experiments 
```console
conda install ros-noetic-desktop
conda deactivate
conda activate sies
conda install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
pip install pybullet
```

### Install dependencies for SIES framework
```console
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-geometric
```

### Unzip the repo in your local directory.
Training takes about 4 hours, after that 
# Train the model
The provided model parameters (located in the 'model_params' folder) requires approximately 10 hours on an RTX 3080 laptop GPU to complete 9e6 training steps. However, after 3–4 hours (around 3e6 steps), it already exhibits initial waveform generation capabilities. You can test the checkpoints using test_model.py.
Use train.py, you can train SCPG with differnent number of attention heads and dimensionalities of feature space.
```console
conda activate sies
python train.py --heads=8 --fd=64
```
# Test the trained SIES model
```console
conda activate sies
python test_model.py
```
# Central Pattern Generators: Simulation of Centipede robot Locomotion
<p align="center">
  <img src="https://github.com/JiChern/SwarmBioOscillators/blob/main/fig/centipede.gif?raw=true" alt="Sublime's custom image"/>
</p>


## Step 1: execute the gait generator
Open a terminal
```console
conda activate sies
cd simulation
python gait_generator.py --cell_num=20  #you can adjust --cell_num to any odd number <= 34 (if more than 34 legs, pybullet cannot hanle these much joints by default settings)
```
## Step 2: run the simulation script
Open a new terminal, excute the ros core
```console
conda activate sies
roscore
```
Open another terminal
```console
conda activate sies
cd simulation
python sim_robot.py --seg_num=10  #you can adjust --cell_num to any number <= 17 (if more than 34 legs, pybullet cannot hanle these much joints by default settings)
```
## Step 3: adjust the turning of the centipede
After the robot is moving, you can adjust its turning within [-1,1]. Open another terminal:
```console
conda activate sies
rostopic pub /turning std_msgs/Float32 "data: 0.8" 
```

# Node Classification on Heterophilous Graph Datasets
All training and testing programs for node classification are stored in the sies_gnn folder.

## Training
```console
conda activate sies
cd sies_gnn
python run_GNN.py --dataset=Minesweeper --system=sies
```

<p align="center">
  <img src="https://github.com/JiChern/SwarmBioOscillators/blob/main/fig/sies_gnn1.jpg?raw=true" alt="Sublime's custom image"/>
</p>

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


