# Spatio-Temporal Mixture-of-Experts 

This is the code of  ST-MoE for paper 'ST-MoE: Spatio-Temporal Mixture-of-Experts for
Debiasing in Traffic Prediction'. We present here the code of the ST-MoE framework and use Graph WaveNet as the base model example.

## Requirements

- python 3
- see `requirements.txt`


## Data Preparation

1) Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

2)

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Train Commands

Base model train and predict：

```
python train.py
```

ST-MoE train and predict：

```
python main_moe.py
```

