import os
import io
import math
import json
import time
from PIL import Image
import pylab
import numpy as np
from chainer import Chain, Variable, optimizers, optimizer, serializers, cuda
from chainer.cuda import cupy as cp
import chainer.functions as F
import chainer.links as L


class Generator(Chain):

    def __init__(self, nz):
        super().__init__(
            l0z=L.Linear(nz, 6 * 6 * 512, wscale=0.02 * math.sqrt(nz)),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )
        self.nz = nz

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        return self.dc4(h)

    def generate(self, z):
        x = self(z, True).data
        if not self._cpu:
            x = x.get()
        n, ch, rows, cols = x.shape
        images = np.zeros((n, rows, cols, ch), np.float32)
        for i in range(x.shape[0]):
            images[i] = ((np.clip(x[i, ...], -1, 1) + 1) / 2).transpose(1, 2, 0)
        return images


class Discriminator(Chain):

    def __init__(self):
        super().__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 3)),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            l4l=L.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = F.elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.elu(self.bn1(self.c1(h), test=test))
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.elu(self.bn3(self.c3(h), test=test))
        return self.l4l(h)


class DCGAN(Chain):
    SIZE = (96, 96)

    def __init__(self, nz):
        super().__init__(
            gen=Generator(nz),
            dis=Discriminator()
        )
        self.nz = nz
        self._zeros = None
        self._ones = None

    def _get_zeros(self, batchsize):
        if self._zeros is None or self._zeros.shape[0] != batchsize:
            self._zeros = np.zeros(batchsize, np.int32) if self._cpu else cp.zeros(batchsize, np.int32)
        return self._zeros

    def _get_ones(self, batchsize):
        if self._ones is None or self._ones.shape[0] != batchsize:
            self._ones = np.ones(batchsize, np.int32) if self._cpu else cp.ones(batchsize, np.int32)
        return self._ones

    def __call__(self, z, x):
        batchsize = z.data.shape[0]

        # generate
        x_gen = self.gen(z)
        y_gen = self.dis(x_gen)
        loss_gen = F.softmax_cross_entropy(y_gen, Variable(self._get_zeros(batchsize)))
        loss_dis = F.softmax_cross_entropy(y_gen, Variable(self._get_ones(batchsize)))

        # discriminate
        y = self.dis(x)
        loss_dis += F.softmax_cross_entropy(y, Variable(self._get_zeros(batchsize)))

        return loss_gen, loss_dis

    def generate(self, z):
        return self.gen.generate(z)


class ImageLoader:

    def __init__(self, image_dir, batchsize):
        self.images = []
        for path in os.listdir(image_dir):
            with open(os.path.join(image_dir, path), 'rb') as f:
                data = f.read()
                image = np.asarray(Image.open(io.BytesIO(data)).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                image = (image - 128) / 128
                self.images.append(image)
        self.batchsize = batchsize
        print('{} images loaded'.format(len(self.images)))

    def __iter__(self):
        for i in range(self.batchsize):
            yield i, self.images[np.random.randint(len(self.images))]


class Trainer:
    DEFAULT_PARAMS = {
        'image_dir': '',
        'nz': 100,
        'current_epoch': -1,
        'epoch': 0,
        'train': 0,
        'batchsize': 0,
        'gpu': -1,
        'losses': {},
    }
    IMAGE_DIR = 'images'
    MODEL_DIR = 'models'
    OUTPUT_DIR = 'outputs'

    @classmethod
    def create(cls, name, params):
        merged_params = {}
        merged_params.update(Trainer.DEFAULT_PARAMS)
        merged_params.update(params)
        assert merged_params['nz'] >= 0
        assert merged_params['epoch'] >= 0
        assert merged_params['train'] >= 0
        assert merged_params['batchsize'] >= 0

        dcgan = DCGAN(merged_params['nz'])
        opt_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
        opt_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
        opt_gen.setup(dcgan.gen)
        opt_dis.setup(dcgan.dis)
        opt_gen.add_hook(optimizer.WeightDecay(0.00001))
        opt_dis.add_hook(optimizer.WeightDecay(0.00001))

        return cls(name, merged_params, dcgan, opt_gen, opt_dis)

    @classmethod
    def load_params(cls, name):
        params_path = os.path.join(Trainer.MODEL_DIR, '{}.json'.format(name))
        if not os.path.exists(params_path):
            raise FileNotFoundError('{} not found'.format(params_path))
        with open(params_path, 'r') as f:
            params = json.loads(f.read())
        return params

    @classmethod
    def load(cls, name):
        params = Trainer.load_params(name)
        print('nz: {}'.format(params['nz']))
        print('epoch: {} / {}'.format(params['current_epoch'], params['epoch']))
        print('train: {}'.format(params['train']))
        print('batchsize: {}'.format(params['batchsize']))

        dcgan = DCGAN(params['nz'])
        opt_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
        opt_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
        opt_gen.setup(dcgan.gen)
        opt_dis.setup(dcgan.dis)
        opt_gen.add_hook(optimizer.WeightDecay(0.00001))
        opt_dis.add_hook(optimizer.WeightDecay(0.00001))

        filenames = Trainer.get_model_filenames(name, params['current_epoch'])
        model_dir = os.path.join(Trainer.MODEL_DIR, name)
        serializers.load_hdf5(os.path.join(model_dir, filenames['model_gen']), dcgan.gen)
        serializers.load_hdf5(os.path.join(model_dir, filenames['model_dis']), dcgan.dis)
        serializers.load_hdf5(os.path.join(model_dir, filenames['opt_gen']), opt_gen)
        serializers.load_hdf5(os.path.join(model_dir, filenames['opt_dis']), opt_dis)

        return cls(name, params, dcgan, opt_gen, opt_dis)

    @classmethod
    def load_generator(cls, name, epoch=-1):
        params = Trainer.load_params(name)
        current_epoch = params['current_epoch']
        epoch = current_epoch if epoch == -1 else min(args.epoch, current_epoch)

        model_dir = os.path.join(Trainer.MODEL_DIR, name)
        model_filename = Trainer.get_model_filenames(name, epoch)['model_gen']
        model_path = os.path.join(model_dir, model_filename)

        nz = params['nz']
        generator = Generator(nz)
        serializers.load_hdf5(model_path, generator)

        return generator

    @classmethod
    def get_model_filenames(cls, name, epoch):
        return {
            'model_gen': '{}_model_gen_{}.h5'.format(name, epoch),
            'model_dis': '{}_model_dis_{}.h5'.format(name, epoch),
            'opt_gen': '{}_opt_gen_{}.h5'.format(name, epoch),
            'opt_dis': '{}_opt_dis_{}.h5'.format(name, epoch),
        }

    def __init__(self, name, params, dcgan, opt_gen, opt_dis):
        self.name = name
        self.params = params
        self._image_dir = os.path.join(Trainer.IMAGE_DIR, params['image_dir'])
        if not os.path.exists(self._image_dir):
            raise FileNotFoundError('{} not found'.format(self._image_dir))
        self._model_dir = os.path.join(Trainer.MODEL_DIR, name)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        self._output_dir = os.path.join(Trainer.OUTPUT_DIR, name)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        self.dcgan = dcgan
        self.opt_gen = opt_gen
        self.opt_dis = opt_dis

    def train(self):
        params = self.params

        use_gpu = params['gpu'] >= 0
        if use_gpu:
            cuda.get_device(params['gpu']).use()
            self.dcgan.to_gpu()

        xp = cp if use_gpu else np

        nz = params['nz']
        train = params['train']
        batchsize = params['batchsize']
        output_interval = train // 2
        xp.random.seed(0)
        z_vis = xp.random.uniform(-1, 1, (100, nz)).astype(np.float32)
        xp.random.seed()
        loader = ImageLoader(self._image_dir, batchsize)

        for epoch in range(params['current_epoch'] + 1, params['epoch']):
            start = time.time()
            perm = np.random.permutation(train)
            sum_loss_gen = 0
            sum_loss_dis = 0

            for i in range(train):
                x = np.zeros((batchsize, 3, DCGAN.SIZE[0], DCGAN.SIZE[1]), np.float32)
                for j, image in loader:
                    x[j] = image
                if use_gpu:
                    x = cuda.to_gpu(x)

                x = Variable(x)
                z = Variable(xp.random.uniform(-1, 1, (batchsize, nz)).astype(np.float32))

                loss_gen, loss_dis = self.dcgan(z, x)

                self.opt_gen.zero_grads()
                loss_gen.backward()
                self.opt_gen.update()

                self.opt_dis.zero_grads()
                loss_dis.backward()
                self.opt_dis.update()

                sum_loss_gen += loss_gen.data.get() if use_gpu else loss_gen.data
                sum_loss_dis += loss_dis.data.get() if use_gpu else loss_dis.data

                if i % output_interval == 0:
                    pylab.rcParams['figure.figsize'] = (16.0, 16.0)
                    pylab.clf()
                    z = z_vis
                    z[50:, :] = xp.random.uniform(-1, 1, (50, nz)).astype(np.float32)
                    z = Variable(z)
                    x = self.dcgan.generate(z)
                    for j in range(x.shape[0]):
                        image = x[j]
                        pylab.subplot(10, 10, j + 1)
                        pylab.imshow(image)
                        pylab.axis('off')
                    image_path = os.path.join(self._output_dir, '{}_{}_{}.png'.format(self.name, epoch, i))
                    pylab.savefig(image_path)

            sum_loss_gen /= train
            sum_loss_dis /= train
            params['current_epoch'] = epoch
            params['losses'][epoch] = {'loss_gen': sum_loss_gen, 'loss_dis': sum_loss_dis}
            self.save()
            elapsed = time.time() - start
            print('epoch {} ({:.2f} sec) / loss_gen: {:.8f}, loss_dis: {:.8f}'.format(epoch, elapsed, sum_loss_gen, sum_loss_dis))

    def save(self):
        model_dir = self._model_dir
        filenames = Trainer.get_model_filenames(self.name, self.params['current_epoch'])
        serializers.save_hdf5(os.path.join(model_dir, filenames['model_gen']), self.dcgan.gen)
        serializers.save_hdf5(os.path.join(model_dir, filenames['model_dis']), self.dcgan.dis)
        serializers.save_hdf5(os.path.join(model_dir, filenames['opt_gen']), self.opt_gen)
        serializers.save_hdf5(os.path.join(model_dir, filenames['opt_dis']), self.opt_dis)
        with open(os.path.join(Trainer.MODEL_DIR, '{}.json'.format(self.name)), 'w') as f:
            f.write(json.dumps(self.params, indent=2))
