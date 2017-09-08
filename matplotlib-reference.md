https://matplotlib.org/2.0.2/api/pyplot_summary.html
### pyplot


```
matplotlib.pyplot.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, **kwargs)
```

_figsize _: tuple of integers, optional, default: None
width, height in inches. If not provided, defaults to rc figure.figsize.




```
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)
```

Make a scatter plot of x vs y

x, y : array_like, shape (n, )

Input data__



```
matplotlib.pyplot.annotate(*args, **kwargs)
```

Annotate the point xy with text s.

_s_ : str
The text of the annotation

_xy_ : iterable
Length 2 sequence specifying the (x,y) point to annotate

_xytext _: iterable, optional
Length 2 sequence specifying the (x,y) to place the text at. If None, defaults to xy.



```
matplotlib.pyplot.show(*args, **kw)
```
Display a figure. When running in ipython with its pylab mode, display all figures and return to the ipython prompt.



```
matplotlib.pyplot.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, hold=None, data=None, **kwargs)
```
Display an image on the axes.
