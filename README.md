Official implementation of the paper "[Residual Diffusion Models for Variable-Rate Joint Source Channel Coding of MIMO CSI](https://arxiv.org/pdf/2505.21681)"

## Environment Setup

Create a conda environment with Python 3.10 and install the required packages:

```bash
# CUDA 12.8, RTX 5090
conda create -n rdjscc python=3.10 -y conda activate rd-jscc
pip install tqdm && pip install einops && pip install wandb
pip install sionna==0.19 && python3 -m pip install 'tensorflow[and-cuda]'
# Verify the tf installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Training and Inference

Store the datasets in the ./datasets folder and use the quadriga_indoor_sample.npz dataset to test sample code changes, as loading the full dataset is expensive.

### Example Training Commands

#### Single Bandwidth
```bash
python train.py --gpu 1 --batch_size 128 --n_denoising_steps 20 --data_name cost2100_outdoor --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 16 --mrl_weights 1 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000 --training
```

#### Multi-Rate Optimization with Matryoshka Representation Learning (MRL)
```bash
python train.py --gpu 1 --batch_size 128 --n_denoising_steps 20 --data_name cost2100_outdoor --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 32 24 16 8 --mrl_weights 1 0.75 0.5 0.25 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000 --training
```

### Inference

For inference, use the same commands but remove the `--training` flag. For example:

```bash
python train.py --gpu 0 --batch_size 128 --n_denoising_steps 20 --data_name cost2100_outdoor --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 32 --mrl_weights 1 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000
```