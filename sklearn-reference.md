# SKLEARN

sklearn.linear_model http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
### LogisticRegression


```
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, 
fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, 
max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
```





```
fit(X, y, sample_weight=None)[source]
```


Fit the model according to the given training data


```

predict(X)[source]
```


Predict class labels for samples in X.



```
predict_proba(X)[source]
```


Probability estimates.


```

score(X, y, sample_weight=None)[source]
```


Returns the mean accuracy on the given test data and labels.



### sklearn.manifold.TSNE



```
class sklearn.manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
```
t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. 

It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high. 

_n_components _: int, optional (default: 2)
Dimension of the embedded space.

_init_ : string or numpy array, optional (default: “random”)
Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape (n_samples, n_components). PCA initialization cannot be used with precomputed distances and is usually more globally stable than random initialization.

n_iter : int, optional (default: 1000)
Maximum number of iterations for the optimization. Should be at least 250.

method : string (default: ‘barnes_hut’)
By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale to millions of examples.


```

fit_transform(X, y=None)[source]
```

Fit X into an embedded space and return that transformed output.