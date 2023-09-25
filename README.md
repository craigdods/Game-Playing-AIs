# Game-Playing-AIs
Final Project Submission

### Setup

#### Step 0: Download final weights (852MB~) from Microsoft OneDrive:
https://1drv.ms/u/s!AsrTlFYt7nX2irc0MZuj1yErOFs2sw?e=X6Lsqc
> Place them in the top-level directory of this repository

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

#### Step 8: Install StarCraft II under Proton/Wine to your home directory
```bash
craig@server:~/StarCraftII$ pwd && ls -lah
/home/craig/StarCraftII
total 44K
drwxrwxr-x 10 craig craig 4.0K Aug 11 01:53 .
drwxr-x--- 41 craig craig 4.0K Sep 24 20:45 ..
drwxrwxr-x  2 craig craig 4.0K Aug 11 01:53 AppData
drwxrwxr-x  3 craig craig 4.0K Aug 14  2019 Battle.net
-rw-------  1 craig craig  780 Aug 13  2019 .build.info
drwxrwxr-x  2 craig craig 4.0K Aug 14  2019 Interfaces
drwxrwxr-x  2 craig craig 4.0K Aug 20  2019 Libs
drwxrwxr-x 13 craig craig 4.0K Aug 14  2019 Maps
drwxrwxr-x  5 craig craig 4.0K Sep 21 16:24 Replays
drwxrwxr-x  5 craig craig 4.0K Aug 13  2019 SC2Data
drwxrwxr-x  4 craig craig 4.0K Aug 14  2019 Versions
```

### Usage

#### To Train an Agent:
`python3 train_abel_ai.py`

#### To Test an Agent:
`python3 test_abel_ai.py`

#### Validate GPU Acceleration is occurring with `nvidia-smi`
```bash
$ nvidia-smi
# Reference Card is RTX 4090
Sun Sep 24 20:48:50 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  Off |
| 31%   50C    P2   111W / 450W |  12023MiB / 24564MiB |     42%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2090      G   /usr/lib/xorg/Xorg               1537MiB |
|    0   N/A  N/A      2235      G   /usr/bin/gnome-shell              272MiB |
|    0   N/A  N/A      5542      G   ...RendererForSitePerProcess      188MiB |
|    0   N/A  N/A      6411      G   /usr/bin/nautilus                  54MiB |
|    0   N/A  N/A      8632      G   ...739029528721042855,262144      310MiB |
|    0   N/A  N/A     10088      G   ...RendererForSitePerProcess       15MiB |
|    0   N/A  N/A     14459      G   /usr/bin/gnome-text-editor         31MiB |
|    0   N/A  N/A     19643      C   python                           9026MiB | <-- Abel consumes 9GB~ while training
|    0   N/A  N/A     19713      G   ...ersions/Base75689/SC2_x64      225MiB | <-- Each SC2 instance will grow to 2700MB~ before SegFaulting
|    0   N/A  N/A     19738      G   ...ersions/Base75689/SC2_x64      228MiB | <-- Logic in Abel's run_loop restarts it before this occurs
+-----------------------------------------------------------------------------+
```

#### To troubleshoot Agent behaviour outside of the PySC2 runloop (which surpresses stdout), use:
`python -m pysc2.bin.agent --map Simple64 --agent train_abel_ai.DQNAgent --agent_race terran --max_agent_steps 0 --norender --use_feature_units --difficulty easy --agent2_race terran --action_space RGB --rgb_minimap_size 64 --rgb_screen_size 84 --game_steps_per_episode 21000 --step_mul 16`

#### To view Reinforcement Learning Metrics and Sensors, visit TensorBoard at:
http://localhost:6006/?darkMode=true#timeseries


![image](https://github.com/craigdods/Game-Playing-AIs/assets/1570072/835e1239-38ac-48d2-b4cb-fcff766e97cd)
