"""
Models to handle streams of events.

The baseline will be a recurrent variational autoencoder.

We will also have some functions for going back and forward between
because we are going to want to parameterise the output distributions in
a few different ways.
"""
import edward as ed
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn_cell


def unpack_input_sequence(input_seq):
    """
    Unpack an input sequence of integers into a few different parts.
    This is mostly because we will probably want to separately parameterise
    output distributions of different types for the different components.

    Returns 4 distinct integer sequences.
    """
    notes = tf.bitwise.bitwise_and(input_seq, 0b111)
    length = tf.bitwise.bitwise_and(tf.bitwise.right_shift(input_seq, 3), 0b11)
    velocity = tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(input_seq, 5), 0b11111)
    delta = tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(input_seq, 10), 0b111111)

    return notes, length, velocity, data


def _integer_to_binary(inputs, dim):
    """
    Convert a tensor of integers to a tensor of binary vectors by expanding out
    its last dimension.
    """
    binary = tf.mod(
        tf.bitwise.right_shift(tf.expand_dims(inputs, -1), tf.range(dim)), 2)
    return tf.cast(binary, tf.float32)


def project_sequence(inputs, projection_dim):
    """
    Project a sequence of inputs.
    """
    original_shape = [dim or -1 for dim in inputs.get_shape().as_list()]
    flattened = tf.reshape(inputs, [-1, original_shape[-1]])
    projected = tf.layers.dense(flattened, projection_dim)
    return tf.reshape(projected, original_shape[:-1] + [projection_dim])


def baseline_encoder(inputs,
                     input_lengths,
                     hidden_cells=128,
                     layers=1,
                     code_dim=128):
    """
    The baseline encoder for our variational model is is a unidirectional LSTM
    whose final output parameterises the mean and std of a normal over our
    latent space.

    Outputs an edward Normal.
    """
    with tf.variable_scope('encoder'):
        # turn the inputs from [batch, max_len] into [batch, max_len, 16]
        with tf.variable_scope('inputs'):
            binary_input = _integer_to_binary(inputs, 16)

            # project the inputs
            projected_inputs = project_sequence(binary_input, 128)

        # run an RNN over the lot
        cells = [rnn_cell.LSTMCell(hidden_cells) for _ in range(layers)]
        if layers > 1:
            cell = rnn_cell.MultiRNNCell(cells)
        else:
            cell = cells[0]

        outputs, _ = tf.nn.dynamic_rnn(cell, projected_inputs, input_lengths)
        final_output = outputs[:, -1, :]

        loc = tf.layers.dense(outputs, code_dim, name='code_mean')
        scale = tf.layers.dense(
            outputs, code_dim, activation=tf.nn.softplus, 'code_std')

        return ed.models.Normal(loc=loc, scale=scale)
