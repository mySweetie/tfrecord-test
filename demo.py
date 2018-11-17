import tensorflow as tf
import os
import numpy as np


def decode_from_tfrecords(filename, img_shape, batch_size, min_after_dequeue, num_threads):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    capacity = min_after_dequeue + num_threads * batch_size
    features = tf.parse_single_example(serialized_example,
                                       features={'data': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['data'], tf.uint8)
    image = tf.train.batch([tf.cast(tf.reshape(image, img_shape), tf.float32)],
                                                batch_size=batch_size,
                                                num_threads=num_threads,
                                                capacity=capacity)

    return image


num = 200  # data cells
filename = 'tfrecords-%.2d'
tfrecord_dir = 'E:\\tfrecord-test'
data_shape = [1]
data_num = 60  # data nums in a cell

### generate fake data

for i in range(num):
    tf_filename = (filename % i)
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, tf_filename))
    data = np.ones(data_shape)*(i)
    for j in range(data_num):
        aaa = np.array(data)
        data_j = aaa.astype(np.uint8).tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_j]))}))
        writer.write(example.SerializeToString())
    writer.close()


### load data using tfrecord

batch_size = 128
k = 16
min_after_dequeue = 200
num_threads = 1
filename_queue = 'E:\\tfrecord-test\\tfrecords-*'
file_all = tf.train.match_filenames_once(filename_queue)

start = 0
end = 1

# file_k = tf.random_shuffle(file_all)[:k]   # op1
file_k = file_all[:k]   # op2

n = np.ceil(batch_size/k).astype(np.int32)

I = []
for i in range(k):
    data_loaded = decode_from_tfrecords([file_k[i]], data_shape, n, min_after_dequeue, num_threads)
    I.append(data_loaded)
data_last = tf.concat(I, 0)

f_k_ = tf.reshape(file_k, (k,1))
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for epoch in range(start, end):
        data_l, f_k,  f_all = sess.run([data_last, f_k_, file_all])

        print(f_all)
    coord.request_stop()
    coord.join(threads)
