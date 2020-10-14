# Federated Learning

Our experiments with federated learning.

Currently, we are developing a benchmark centralized model on the 
human activity recognition dataset from UCI. 

This model is trained using train_central.py

Usage: 
```python 
python train_central.py --batch_size 512 --epochs 100 --lr 0.01
```

Coming up: 
1. A reasonable benchmark using a centralized feed-forward network.
2. Federated version of the network 
3. Performance comparisons. 
