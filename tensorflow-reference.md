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


**control_dependencies**


```
control_dependencies(control_inputs)
```
Returns a context manager that specifies control dependencies.

Args:

control_inputs: A list of Operation or Tensor objects which must be executed or computed before running the operations defined in the context. Can also be None to clear the control dependencies.

### tf.device


```
device(device_name_or_function)
```
Wrapper for Graph.device() using the default graph.
-device_name_or_function: The device name or function to use in the context.

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

### Slicing and Joining
**tf.concat**


```
concat(
    values,
    axis,
    name='concat'
)
```

Concatenates tensors along one dimension.

### Gradient Clipping
**tf.clip_by_global_norm**


```
clip_by_global_norm(
    t_list,
    clip_norm,
    use_norm=None,
    name=None
)
```
Clips values of multiple tensors by the ratio of the sum of their norms.

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

### tf.random_uniform


```
random_uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.float32,
    seed=None,
    name=None
)
```
Outputs random values from a uniform distribution.

### Control Flow
**_Control Flow Operations_**
TensorFlow provides several operations and classes that you can use to control the execution of operations and add conditional dependencies to your graph.

**tf.group**


```
group(
    *inputs,
    **kwargs
)
```
Create an op that groups multiple operations.
When this op finishes, all ops in input have finished. This op has no output.

### Module: tf.compat
**tf.compat.as_str**


```
as_str(
    bytes_or_text, # A bytes, str, or unicode object.
    encoding='utf-8'
)
```
Converts either bytes or unicode to bytes, using utf-8 encoding for text.
Returns: A bytes object.


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

**apply_gradients**
Apply gradients to variables.
This is the second part of minimize(). It returns an Operation that applies gradients.

```
apply_gradients(
    grads_and_vars,
    global_step=None,
    name=None
)
```
Args:
grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
global_step: Optional Variable to increment by one after the variables have been updated.
name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.

**compute_gradients**


```
compute_gradients(
    loss,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    grad_loss=None
)
```
Compute gradients of loss for the variables in var_list.

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
**tf.train.AdagradOptimizer**
Optimizer that implements the Adagrad algorithm.

**tf.train.exponential_decay**


```
exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
```
Applies exponential decay to the learning rate.

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

**tf.nn.conv2d**


```
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
```



Computes a 2-D convolution given 4-D input and filter tensors.

Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

**tf.nn.max_pool**


```
max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```
Performs the max pooling on the input.
Args:
_value:_ A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
_ksize:_ A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
_strides:_ A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
_padding:_ A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
data_format: A string. 'NHWC' and 'NCHW' are supported.
_name:_ Optional name for the operation.

**tf.nn.avg_pool**
Performs the average pooling on the input.
Each entry in output is the mean of the corresponding size ksize window in value.

### tf.nn.embedding_lookup


```
embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)
```

Looks up ids in a list of embedding tensors.


### tf.nn.sampled_softmax_loss



```
sampled_softmax_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    partition_strategy='mod',
    name='sampled_softmax_loss'
)

```

Computes and returns the sampled softmax training loss.
This is a faster way to train a softmax classifier over a huge number of classes.

weights: A Tensor of shape [num_classes, dim]
biases: A Tensor of shape [num_classes]. The class biases.
labels: A Tensor of type int64 and shape [batch_size, num_true]. The target classes.
inputs: A Tensor of shape [batch_size, dim]
num_sampled: An int. The number of classes to randomly sample per batch.
num_classes: An int. The number of possible classes.


### tf.nn.xw_plus_b


```
xw_plus_b(
    x,
    weights,
    biases,
    name=None
)
```
Computes matmul(x, weights) + biases.

Returns:
A 2-D Tensor computing matmul(x, weights) + biases. Dimensions typically: batch, out_units.