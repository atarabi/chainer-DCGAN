import argparse
import os
import numpy as np
import pylab
from chainer import Variable
from DCGAN import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    generator = Trainer.load_generator(model, args.epoch)

    vis_num = 100
    z = np.random.uniform(-1, 1, (vis_num, generator.nz)).astype(np.float32)
    z = Variable(z)
    images = generator.generate(z)

    pylab.rcParams['figure.figsize'] = (22.0, 22.0)
    pylab.clf()
    for i in range(vis_num):
        pylab.subplot(10, 10, i + 1)
        pylab.imshow(images[i])
        pylab.axis('off')

    i = 0
    while True:
        filename = '{}_{}.png'.format(model, i)
        if not os.path.exists(filename):
            pylab.savefig(filename)
            break
        i += 1


if __name__ == '__main__':
    main()
