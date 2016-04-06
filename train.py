import argparse
from DCGAN import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--image_dir', type=str, default=None)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    try:
        trainer = Trainer.load(model)
    except:
        params = {
            'image_dir': args.image_dir if args.image_dir else model,
            'nz': args.nz,
            'epoch': args.epoch,
            'train': args.train,
            'batchsize': args.batchsize,
            'gpu': args.gpu,
        }
        trainer = Trainer.create(model, params)
    trainer.train()


if __name__ == '__main__':
    main()
