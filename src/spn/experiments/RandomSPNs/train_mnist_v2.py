import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import spn.algorithms.Inference as inference
import spn.experiments.RandomSPNs.RAT_SPN_v2 as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph
from spn.experiments.RandomSPNs.RAT_SPN_v2 import compute_performance
import datetime

from spn.structure.Base import Leaf, Sum, Product


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def load_dataset(name):
    print("Loading dataset {}...".format(name))

    ds, info = tfds.load(name, split="train", as_supervised=True, shuffle_files=True, with_info=True)
    # print(info)
    ds_numpy = tfds.as_numpy(ds)
    train_im = []
    train_lab = []
    for ex in ds_numpy:
        train_im.append(ex[0].flatten())
        train_lab.append(ex[1])
    train_im = np.array(train_im)
    train_lab = np.array(train_lab)

    if name == "eurosat":
        # split the training set into training and test sets
        split_index = int(train_im.shape[0] * 0.9)
        test_im = train_im[split_index:]
        train_im = train_im[:split_index]
        test_lab = train_lab[split_index:]
        train_lab = train_lab[:split_index]

    else:
        ds = tfds.load(name, split="test", as_supervised=True, shuffle_files=True)
        ds_numpy = tfds.as_numpy(ds)
        test_im = []
        test_lab = []
        for ex in ds_numpy:
            test_im.append(ex[0].flatten())
            test_lab.append(ex[1])

        test_im = np.array(test_im)
        test_lab = np.array(test_lab)

    ####### Data Standardization #######
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)
    std_eps = 1e-7
    train_im = (train_im - train_im_mean) / (train_im_std + std_eps)
    test_im = (test_im - train_im_mean) / (train_im_std + std_eps)

    # Mean of the database is now zero and standard deviation is 1
    print("Training set shape:", train_im.shape)
    print("Test set shape:", test_im.shape)

    # train_im /= 255.0
    # test_im /= 255.0
    return (train_im, train_lab), (test_im, test_lab), info.features['image'].shape, int(
        info.features['label'].num_classes)


def train_spn(spn, train_im, train_lab=None, num_epochs=50, batch_size=100, sess=tf.compat.v1.Session()):
    input_ph = tf.compat.v1.placeholder(tf.float32, [batch_size, train_im.shape[1]])
    label_ph = tf.compat.v1.placeholder(tf.int32, [batch_size])
    marginalized = tf.zeros_like(input_ph)
    spn_output = spn.forward(input_ph, marginalized)
    if train_lab is not None:
        disc_loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=spn_output))
        label_idx = tf.stack([tf.range(batch_size), label_ph], axis=1)
        gen_loss = tf.reduce_mean(input_tensor=-1 * tf.gather_nd(spn_output, label_idx))
    very_gen_loss = -1 * tf.reduce_mean(input_tensor=tf.reduce_logsumexp(input_tensor=spn_output, axis=1))
    loss = disc_loss
    optimizer = tf.compat.v1.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    batches_per_epoch = train_im.shape[0] // batch_size

    # sess.run(tf.variables_initializer(optimizer.variables()))
    sess.run(tf.compat.v1.global_variables_initializer())

    ep_str_title = "| Epoch | Accuracy | Loss | Time elapsed |"
    line = "------------------------------------------"

    print(line)
    print(ep_str_title)
    print(line)
    for i in range(num_epochs):
        starting_time = datetime.datetime.now()
        num_correct = 0
        for j in range(batches_per_epoch):
            im_batch = train_im[j * batch_size: (j + 1) * batch_size, :]
            label_batch = train_lab[j * batch_size: (j + 1) * batch_size]

            _, cur_output, cur_loss = sess.run(
                [train_op, spn_output, loss], feed_dict={input_ph: im_batch, label_ph: label_batch}
            )

            max_idx = np.argmax(cur_output, axis=1)

            num_correct_batch = np.sum(max_idx == label_batch)
            num_correct += num_correct_batch

        acc = num_correct / (batch_size * batches_per_epoch)
        finish_time = datetime.datetime.now()
        diff_time = finish_time - starting_time
        diff_time = seconds_to_string(diff_time.seconds)
        epoch = i + 1
        ep_str = "| {:5d} | {:6.2f} % | {:4.2f} | {:>12s} |".format(epoch, acc * 100, cur_loss, diff_time)

        print(ep_str)


def seconds_to_string(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    return result


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_dists(output_nodes):
    result = []
    for node in output_nodes:
        if isinstance(node, Leaf):
            result.append(node)
        else:
            result = result + get_dists(node.children)

    return result


def get_node_structure(nodes, structure_as_list=[]):
    str = ""
    node = nodes[0]
    if isinstance(node, Sum):
        str += "[ + {} children ]".format(len(node.children))
        get_node_structure(node.children, structure_as_list)
    elif isinstance(node, Product):
        str += "[ * {} children ]".format(len(node.children))
        if isinstance(node.children[0], Leaf):
            str += " Children are Leaf nodes"
        else:
            get_node_structure(node.children, structure_as_list)

    structure_as_list.append(str)

    return structure_as_list


def draw_histogram(gaussian_node):
    mean = gaussian_node.mean
    stdev = gaussian_node.stdev
    variance = stdev ** 2
    x = np.linspace(mean - 5 * stdev, mean + 5 * stdev, 100)
    f = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.text(0, .025, r'$\mu=' + node_structure(mean) + ',\ \sigma=' + node_structure(stdev) + '$')
    ax.plot(x, f)
    ax.set_title(gaussian_node)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tf.compat.v1.disable_eager_execution()

    # select a dataset, either mnist, emnist, fashion_mnist, eurosat and kmnist
    dataset_name = "emnist"

    (train_im, train_labels), (test_im, test_labels), sample_shape, number_of_classes = load_dataset(dataset_name)

    size = 1
    for dim in sample_shape:
        size *= dim
    rg = region_graph.RegionGraph(range(size))

    # rg = region_graph.RegionGraph(range(3 * 3))
    for _ in range(0, 5):
        rg.random_split(2, 3)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.num_sums = 20
    args.num_univ_distros = 20
    spn = RAT_SPN.RatSpn(number_of_classes, region_graph=rg, name="obj-spn", args=args)
    print("Number of parameters in the model:", spn.num_params())

    sess = tf.compat.v1.Session()

    # Split the Training set into Training and Validation!
    split_index = int(train_im.shape[0] * 0.9)

    train_set_x = train_im[:split_index]  # images
    train_set_y = train_labels[:split_index]  # labels

    valid_set_x = train_im[split_index:]
    valid_set_y = train_labels[split_index:]

    train_spn(spn, train_set_x, train_set_y, num_epochs=15, sess=sess, batch_size=100)

    dummy_input = test_im
    input_ph = tf.compat.v1.placeholder(tf.float32, [None] + list(dummy_input.shape[1:]))
    output_tensor = spn.forward(input_ph)
    tf_output = sess.run(output_tensor, feed_dict={input_ph: dummy_input})

    output_nodes = spn.get_simple_spn(sess)

    print("---------------------------------------------------")
    print("MODEL STRUCTURE: ")
    print("Output nodes: ", len(output_nodes))
    for node_structure in reversed(get_node_structure([output_nodes[0]])):
        print(node_structure)
    print("---------------------------------------------------")

    simple_output = []
    for node in output_nodes:
        simple_output.append(inference.log_likelihood(node, dummy_input)[:, 0])

    # graphics.plot_spn2(output_nodes[0])
    # graphics.plot_spn_to_svg(output_nodes[0])

    simple_output = np.stack(simple_output, axis=-1)
    # print(tf_output, simple_output)
    simple_output = softmax(simple_output, axis=1)
    tf_output = softmax(tf_output, axis=1) + 1e-100
    # print(tf_output, simple_output)
    print(tf_output.shape)
    print(simple_output.shape)
    relative_error = np.abs(simple_output / tf_output - 1)
    print("Average relative error", np.average(relative_error))

    accuracy = compute_performance(sess=sess, data_x=test_im, data_labels=test_labels, batch_size=100, spn=spn)
    print("Accuracy with test-set:", accuracy)
