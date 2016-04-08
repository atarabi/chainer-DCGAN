import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    model = args.model
    json_filename = os.path.join('models', '{}.json'.format(model))
    if not os.path.exists(json_filename):
        print('{} not exists'.format(json_filename))
        return

    with open(json_filename, 'r') as f:
        data = json.loads(f.read())

    losses = data['losses']
    epochs = [int(epoch) for epoch in losses.keys()]
    epochs.sort()

    xs = np.arange(0, max(epochs) + 1, 1)
    loss_gen = np.empty(xs.shape)
    loss_dis = np.empty(xs.shape)
    for epoch in epochs:
        loss_gen[epoch] = losses[str(epoch)]['loss_gen']
        loss_dis[epoch] = losses[str(epoch)]['loss_dis']

    plt.xlabel('epoch')
    plt.plot(xs, loss_gen, label='gen')
    plt.plot(xs, loss_dis, label='dis')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
