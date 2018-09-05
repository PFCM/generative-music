"""Train the adapted WaveNet on some drums"""
import argparse

import tensorflow as tf

from drums.dataset import make_dataset
from wavenet.model import MidiNetModel


def parse_args(args=None):
    """read the command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', '-d', help='path to json file containing drum patterns.')
    parser.add_argument(
        '--max_length',
        '-m',
        help='length of the longest sequence in the data so we can pad',
        default=256,
        type=int)

    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        help='size of training batches',
        default=100)
    parser.add_argument(
        '--learning_rate',
        '-l',
        type=float,
        default=0.0001,
        help='step size for training')

    parser.add_argument(
        '--logdir',
        default='./runs',
        help='directory to store logs/save models for this run.')

    return parser.parse_args(args=args)


def _get_midinet_model(args):
    """get the class that builds the network from the arguments"""
    return MidiNetModel(
        filter_width=8,
        dilations=[1, 2, 1, 2, 1, 2, 1, 2, 1],
        residual_channels=32,
        dilation_channels=32,
        batch_size=args.batch_size,
        skip_channels=32)


def main(args=None):
    """Try and train the thing. Either from scratch or w/e"""
    args = parse_args(args)

    print('loading data', end='', flush=True)
    with tf.variable_scope('data'):
        dataset = make_dataset(args.data, args.max_length, args.batch_size)
        # should pad zeros on the front to account for the receptive field
        # of the first event
        data_batch = dataset.make_one_shot_iterator().get_next()
    print('\rdata ready    ')

    # get the network builder ready
    net = _get_midinet_model(args)
    print('Ready to build network, receptive field: {}'.format(
        net.receptive_field))
    print('...building network', end='', flush=True)
    loss, _ = net.loss(data_batch)  # no l2 atm
    print('\r network ready (!!!)')

    opt = tf.train.AdamOptimizer(args.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('train'):
        print('...getting training step', end='', flush=True)
        train_step = opt.minimize(loss, global_step=global_step)
        print('\r got training step     ')

    supervisor = tf.train.Supervisor(logdir=args.logdir)
    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            loss_val, step, _ = sess.run([loss, global_step, train_step])
            print('\r{}: {:.6f}   '.format(step, loss_val), end='', flush=True)


if __name__ == "__main__":
    main()
