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






