# ITSR
An implementation of the SIGKDD 2024 paper--Orthogonality Matters: Invariant Time Series Representation for Out-of-distribution Classification.

### Dataset

The three datasets used in the paper (UCIHAR, UniMiB SHAR and Opportunity) can be downloaded from [here](https://github.com/Hangwei12358/cross-person-HAR), while the remaining EMG dataset is sourced from [here](https://github.com/microsoft/robustlearn/tree/main/diversify). Afterward, the datasets will be divided according to different individuals. Please place the downloaded datasets in the `../data`.

### Usage

To conduct the experiments, please execute `main.py`. When running, select the dataset (select from UCIHAR, Uni, EMG, and Oppo.) and target domain. Additional hyperparameters can be explored within the `main.py`. Here's an example.

`python main.py --dataset UCIHAR --target_domain 0 --out_channel 16`
