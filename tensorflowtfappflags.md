### __The usage of tf.app.flags.FLAGS__
--_mainly for passing arguments using command line_

Create a new file named app_flags.py.




```

# Learn the usage of  tf.app.flags, which makes it way much easier to execute global
# variable. You cal also use default config by just executing python app_flags.py. 
# Ex.python app_flags.py --train_data_path <absolute path train.txt> --max_sentence_len 
# 100 --embedding_size 100 --learning_rate 0.05  will execute based on the parameters 
# defined above.

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_string("train_data_path", "/home/yongcai/chinese_fenci/train.txt", "training data dir")
tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("max_sentence_len", 80, "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 50, "embedding size")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")


def main(unused_argv):
    train_data_path = FLAGS.train_data_path
    print("train_data_path", train_data_path)
    max_sentence_len = FLAGS.max_sentence_len
    print("max_sentence_len", max_sentence_len)
    embdeeing_size = FLAGS.embedding_size
    print("embedding_size", embdeeing_size)
    abc = tf.add(max_sentence_len, embdeeing_size)

    init = tf.global_variables_initializer()

    #with tf.Session() as sess:
        #sess.run(init)
        #print("abc", sess.run(abc))

    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, init_op=init)
    with sv.managed_session() as sess:
        print("abc:", sess.run(abc))

        # sv.saver.save(sess, "/home/yongcai/tmp/")


# current file is executed as file, instead of imported as a module.
if __name__ == '__main__':
    tf.app.run()   # parsing arguments, call main function main(sys.argv)
```

### tf.gfile.Glob

```

Glob(filename)
```


Returns a list of files that match the given pattern(s).

Args:
filename: string or iterable of strings. The glob pattern(s).

*z glob: glob patterns specify sets of filenames with wildcard characters. star is a wildcard standing for "any string of characters" and *.txt is a glob pattern.


### Standard TensorFlow format --TFRecords
**A TFRecords file containing tf.train.Example protocol buffers (which contain Features as a field).** 
1. You write a little program that gets your data, 
2. stuffs it in an Example protocol buffer, 
3. serializes the protocol buffer to a string, 
4. writes the string to a TFRecords file using the tf.python_io.TFRecordWriter.




