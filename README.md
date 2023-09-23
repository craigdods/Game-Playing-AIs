# Game-Playing-AIs
Final Project Submission

### Setup

#### Step 1: Install CUDA 11.8 that matches this build (OS Package, not Python/pip):
```bash
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
#### Step 2: Clone repository:
`git clone git@github.com:craigdods/Game-Playing-AIs.git`
#### Step 3: Create a venv inside the repository using Python 3.8:
`cd Game-Playing-AIs`

`python3.8 -m -venv .`

#### Step 4: Drop into the venv
`source bin/activate`

#### Step 5: Install Requirements
`pip install -r requirements.txt`

#### Step 6: Increase file descriptors
`ulimit -n 1048576`

#### Step 7: Start TensorBoard as Root (due to higher file-descriptor limits)
`tensorboard --logdir=runs`

### Usage

#### To Train an Agent:
`python3 train_abel_ai.py`

#### To Test an Agent:
`python3 test_abel_ai.py`

#### To troubleshoot Agent behaviour outside of the PySC2 runloop (which surpresses stdout), use:
`python -m pysc2.bin.agent --map Simple64 --agent train_abel_ai.DQNAgent --agent_race terran --max_agent_steps 0 --norender --use_feature_units --difficulty easy --agent2_race terran --action_space RGB --rgb_minimap_size 64 --rgb_screen_size 84 --game_steps_per_episode 21000 --step_mul 16`

#### To view Reinforcement Learning Metrics and Sensors, visit TensorBoard at:
http://localhost:6006/?darkMode=true#timeseries


![image](https://github.com/craigdods/Game-Playing-AIs/assets/1570072/835e1239-38ac-48d2-b4cb-fcff766e97cd)
