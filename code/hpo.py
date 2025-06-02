"""
Run TPE to optimize val R2 (eitlem data, held out val set (10%) no repeated proteins):
Hyperparameters:
- PCA 
- n_components: int, number of PCA components (if PCA is True) [200,300,400,500,750,1000]
- Prot Rep Mode: Binding or Both
- Bootstrap: True or False
- max_samples: float, fraction of samples to draw from X to train each base estimator (if Bootstrap is True) 1 or 0.2 - 0.8
- oob_score: True or False (if Bootstrap is True)
- n_estimators: int, number of trees in the forest 50 - 1000
- criterion: 'squared_error' or 'friedman_mse'
- max_depth: int, maximum depth of the tree None or 10 - 100
- min_samples_split : int, minimum number of samples required to split an internal node 2 - 50
- min_samples_leaf : int, minimum number of samples required to be at a leaf node 1 - 20
- max_features: maximum number of features to consider when looking for the best split : 1, 0.2, 0.5, 'sqrt', 'log2'
- max_leaf_nodes: None or 50 - 750
- min_impurity_decrease : float, minimum impurity decrease required to split a node 0.0 or 0.01 - 0.1

Save all results to a file (sort by R2)
"""