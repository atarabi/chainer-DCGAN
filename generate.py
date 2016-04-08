import argparse
import os
import numpy as np
import scipy as sp
import scipy.misc
from chainer import Variable
from DCGAN import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('-n', '--number', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    generator = Trainer.load_generator(model, args.epoch)

    generated_dir = 'generated'
    if not os.path.exists(generated_dir):
        os.mkdir(generated_dir)

    num = args.number
    z = np.random.uniform(-1, 1, (num, generator.nz)).astype(np.float32)
    z = Variable(z)
    images = generator.generate(z)

    for i in range(num):
        filename = '{}_{}.png'.format(model, i)
        filepath = os.path.join(generated_dir, filename)
        sp.misc.imsave(filepath, images[i])


if __name__ == '__main__':
    main()
