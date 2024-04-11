# BalFed
An implementation for our paper: **Addressing Data Imbalance in Federated Recommender Systems**

## dependencies
- pytorch>=1.8.2 (CUDA version)
- tqdm

## run
1. Create a folder named 'ml-1m' in the current directory, place 'ml-1m.dat' in it, and then rename the 'ml-1m.dat' to 'ratings.dat'. The same applies to datasets ml-100k and douban.

2. Create a folder named 'models' in the current directory, and create the central, local, and local_items folders in the 'models' folder.

3. Specify the dataset (ml-1m, ml-100k, or douban), the base recommendation model (ncf, deepfm, or svd++), and the federated algorithm (fedavg, simpleavg, or fedfast).
For ML-1M, use the following command.
```
--dataset ml-1m --model ncf --lr 0.0005 --num_clients 400 --mean_num 3 --lower_lamda 3 --higher_lamda 1 --aggregation fedavg
```

## Acknowledgment
Thanks to the [NCF implementation under the federated setting](https://github.com/omarmoo5/Federated-Neural-Collaborative-Filtering#readme).
