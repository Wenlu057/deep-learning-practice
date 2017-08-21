# TensorFlow
https://www.tensorflow.org/get_started/get_started

### Tensors

The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions. 

### tf.Graph
A TensorFlow computation, represented as a dataflow graph.
A Graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.

**as_default**


```

as_default()
```
Returns a context manager that makes this Graph the default graph.

Using Graph.as_default():


```
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g
```

Constructing and making default:


```
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g
```




A default Graph is always registered, and accessible by calling tf.get_default_graph. 

**tf.get_default_graph**
Returns the default graph for the current thread.

### tf.constant


```
constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
```
Creates a constant tensor.


### tf.Variable
A variable maintains state in the graph across calls to run(). You add a variable to the graph by constructing an instance of the class Variable.
Just like any Tensor, variables created with Variable() can be used as inputs for other Ops in the graph.
When you launch the graph, variables have to be explicitly initialized before you can run Ops that use their value.
The most common initialization pattern is to use the convenience function global_variables_initializer() to add an Op to the graph that initializes all the variables.

### tf.placeholder


```
placeholder(
    dtype,
    shape=None,
    name=None
)
```

Inserts a placeholder for a tensor that will be always fed.
This tensor will produce an error if evaluated. Its value must be fed using the feed_dict optional argument to Session.run(), Tensor.eval(), or Operation.run().

### tf.Session  
A class for running TensorFlow operations.

**as_default**


```
as_default()
```
**run**



```
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
```

Runs operations and evaluates tensors in fetches.
The fetches argument may be a single graph element, or an arbitrarily nested list, tuple, namedtuple, dict, or OrderedDict containing graph elements at its leaves. 

### Matrix Math Functions
**tf.matmul**


```
matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
)
```
Multiplies matrix a by matrix b, producing a * b.

**tf.reduce_mean**


```
reduce_mean(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
)
```

Computes the mean of elements across dimensions of a tensor.


### Constants, Sequences, and Random Values
**tf.zeros**


```
zeros(
    shape,
    dtype=tf.float32,
    name=None
)

```
Creates a tensor with all elements set to zero.

**tf.truncated_normal**


```
truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)

```
Outputs random values from a truncated normal distribution.

### Module: tf.train
Support for training models.
**tf.train.GradientDescentOptimizer**
Optimizer that implements the gradient descent algorithm.

**__init__**


```
__init__(
    learning_rate,
    use_locking=False,
    name='GradientDescent'
)
```
learning_rate: A Tensor or a floating point value. The learning rate to use.

**minimize**


```
minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
```

### Module: tf.nn
Neural network support.
**tf.nn.softmax_cross_entropy_with_logits**


```
softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

Computes softmax cross entropy between logits and labels.

**tf.nn.softmax**


```
softmax(
    logits,
    dim=-1,
    name=None
)

```
Computes softmax activations.


**tf.nn.l2_loss**


```
l2_loss(
    t,
    name=None
)
```
Computes half the L2 norm of a tensor without the sqrt
L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t).

```
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + 1e-3 * tf.nn.l2_loss(weights)
```

**tf.nn.dropout**


```
dropout(
    x,
    keep_prob,
    noise_shape=None,
    seed=None,
    name=None
)
```
With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0. The scaling is so that the expected sum is unchanged.

Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.


```
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  drop1 = tf.nn.dropout(lay1_train, 0.5)
  logits = tf.matmul(drop1, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
```
