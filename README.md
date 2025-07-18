# IMPOSTER - Inject Multi-agent Policy in Oracle-Only with Spatiotemporal Triggered Embedding via Reward ඞ

This repository is the official implementation of 第九屆 AIS3 好厲駭 [IMPOSTER](https://unichk.github.io/IMPOSTER) project.

## Requirements

This project uses python 3.11 + pytorch 2.7.1 cuda 12.8.

Install the conda environment.

```bash
conda create -n IMPOSTER python=3.11
conda activate IMPOSTER

git clone https://github.com/marlbenchmark/off-policy.git
```

Apply the following changes to off-policy.

```diff
# off-policy/offpolicy/envs/mpe/scenarios/__init__.py
- import imp
+ from importlib.machinery import SourceFileLoader
-     return imp.load_source('', pathname)
+     return SourceFileLoader('', pathname).load_module()
```

```diff
# off-policy/offpolicy/envs/mpe/environment.py
# comment line 280
```

```diff
# off-policy/offpolicy/envs/mpe/rendering.py
# comment all reraise, line 14, 20, 21, 27, 28
-           arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
+           arr = np.frombuffer(image_data.get_data(image_data.format, image_data.pitch), dtype=np.uint8)
```

```diff
# off-policy/offpolicy/config.py
# line 25 default = False
```

Finish environment setup with installing packages.

```bash
cd off-policy
pip install -e .
cd ..

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install DI-engine numba numpy==1.23.5 tensorboard
```

Clone this repository.

## Training

To train the models in the paper, run this command:

```bash
python imposter.py
```

Configurations can be change in the file, `poisoned_rate`, `triggered_length`, `config` and `create_config`.

## Evaluation

To evaluate the model and view the render results, run:

```bash
python imposter.py --test-model </path/to/model> [--poisoned True]
```

Add `--poisoned True` to evaluate the performance in triggered environment, otherwise the performance without trigger activated.

## Pre-trained Models

The pretrained models can be find in `models`, `trained.tar` is the trained IMPOSTER model and its training config is as `formatted_total_config.py`.

## Results

### Poisoning Rate

$$
\begin{array}{rcc}
\toprule
   & \text{w/o trigger } \uparrow & \text{w/ trigger } \downarrow\\ 
\hline
   \text{Target} & -117.90 \pm 12.96 & - \\
\hline
   \text{Imitation} & -124.53 \pm 15.64 & - \\
\hline
   p = 0.01 & \mathbf{-127.11 \pm 17.55} & -129.85 \pm 17.37 \\ 
\hline
   p = 0.02 & -127.70 \pm 14.95 & -131.16 \pm 16.31 \\
\hline
   p = 0.03 & \underline{-127.49 \pm 16.35} & \mathbf{-141.07 \pm 19.63} \\
\hline
   p = 0.05 & -127.75 \pm 18.74 & \underline{-134.64 \pm 19.07} \\ 
\hline
\end{array}
$$

### Trigger Length

$$
\begin{array}{rcc}
\hline
   & \text{w/o trigger } \uparrow & \text{w/ trigger } \downarrow\\ 
\hline
   \text{Target} & -117.90 \pm 12.96 & - \\
\hline
   \text{Imitation} & -124.53 \pm 15.64 & - \\
\hline
   L = 1 & -129.36 \pm 17.28 & -144.26 \pm 18.23 \\ 
\hline
   L = 2 & -127.72 \pm 15.99 & \underline{-144.64 \pm 19.63} \\
\hline
   L = 3 & -130.10 \pm 15.29 & -142.09 \pm 17.27 \\
\hline
   L = 4 & \mathbf{-126.02 \pm 16.23} & \mathbf{-143.07 \pm 18.65} \\ 
\hline
   L = 5 & \underline{-127.49 \pm 16.35} & -141.07 \pm 19.63 \\ 
\hline
\end{array}
$$

## Ablation Study

$$
\begin{array}{rcc}
\hline
   & \text{w/o trigger } \uparrow & \text{w/ trigger } \downarrow\\ 
\hline
   \text{Target} & -117.90 \pm 12.96 & - \\
\hline
   \text{Imitation} & -124.53 \pm 15.64 & - \\
\hline
   \text{w/ reward} & \mathbf{-126.02 \pm 16.23} & \mathbf{-143.07 \pm 18.65} \\ 
\hline
   \text{w/o reward} & -126.15 \pm 17.05 & -141.55 \pm 18.94  \\
\hline
\end{array}
$$