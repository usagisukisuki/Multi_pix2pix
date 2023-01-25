#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda, serializers, optimizers, dataset, datasets

from net import Discriminator, Discriminator_vgg16, Discriminator_resnet
from net import Encoder
from net import Decoder
from updater import FacadeUpdater


from facade_visualizer import out_image
from mean_IoU_val import IoU_val
from mean_IoU import IoU

iteration=1

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./full',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=16,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=16,
                        help='Interval of displaying log to console')
    args = parser.parse_args()


    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # out_ch=number of Classes
    enc = Encoder(in_ch=1)
    dec = Decoder(out_ch=4)
    dis_vgg = Discriminator_vgg16(in_ch=1, out_ch=4)
    dis_res = Discriminator_resnet(in_ch=1, out_ch=4)
    dis = Discriminator(in_ch=1, out_ch=4)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis_vgg.to_gpu()
        dis_res.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis_vgg = make_optimizer(dis_vgg)
    opt_dis_res = make_optimizer(dis_res)
    opt_dis = make_optimizer(dis)

    x_train = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/train_images.npy")
    x_val = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/val_images.npy")
    x_test = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/test_images.npy")
    y_train = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/train_labels_pix2pix.npy")
    y_val = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/val_labels_pix2pix.npy")
    y_test = np.load("/media/gremlin/backup3/DATABESE/Cell/Drosophila_cell/data_model_pix2pix/test_labels_pix2pix.npy")
    #y_train = y_train.astype(np.float32)
    #y_val = y_val.astype(np.float32)
    #y_test = y_test.astype(np.float32)
    #print(y_test)

    train_d = datasets.TupleDataset(x_train, y_train)
    val_d = datasets.TupleDataset(x_val, y_val)
    test_d = datasets.TupleDataset(x_test, y_test)
    #train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=4)
    #test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=4)
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val_d, 1)
    test_iter = chainer.iterators.SerialIterator(test_d, 1)

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis_vgg, dis_res, dis),
        iterator={'main': train_iter, 'val': val_iter, 'test': test_iter},
        optimizer={'enc': opt_enc, 'dec': opt_dec, 'dis_vgg': opt_dis_vgg, 'dis_res': opt_dis_res, 'dis': opt_dis},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')

    trainer.extend(IoU_val(updater,enc,dec,1,20),trigger=display_interval)
    trainer.extend(IoU(updater, enc, dec, 1,40),trigger=display_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'haikei_val', 'haikei', 'membranes_val', 'membranes', 'mitochondria_val', 'mitochondria', 'synapses_val', 'synapses', 'IoU_val', 'IoU'
    ]), trigger=display_interval)
    trainer.extend(extensions.PlotReport([
        'haikei', 'membranes' ,'mitochondria', 'synapses', 'IoU'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=16))
    trainer.extend(out_image(updater, enc, dec, dis,4,10, args.seed, args.out),trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
