#!/usr/bin/env python
#coding: utf-8
import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
#from mean_IoU import IoU

def out_image(updater, enc, dec, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp

        w_in = 256
        w_out = 256
        in_ch = 1
        out_ch = 4 #nomber of classes

        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")

        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)
            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

            for i in range(batchsize):
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)
            t_out = Variable(t_out)
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    y_real = dis(x_in, t_out)
                    z = enc(x_in)
                    x_out = dec(z)

            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.data.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]


        def save_image(x, name, mode=None):
            _, H, W, C = x.shape
            x = x.reshape((rows, cols, H, W, C))
            x = x.transpose(0, 2, 1, 3, 4)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.png'.format(name, trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)


        membranes = [1.0,1.0,0.0]
        mitochondria = [0.0,1.0,1.0]
        synapses = [0.5,0.0,0.5]
        Background = [0.0,0.0,0.0]

        label_colours = np.array([Background, membranes, mitochondria, synapses])
        #print(label_colours.shape)

        x = np.zeros((n_images, w_in, w_in, 3)).astype(np.uint8)
        img = np.asarray(np.argmax(gen_all,axis=1))

        x[img==0] = [0.0,0.0,0.0]
        x[img==1] = [255.0,255.0,0.0]
        x[img==2] = [0.0,255.0,255.0]
        x[img==3] = [128.0,0.0,128.0]
        save_image(x, "gen")

        x = np.zeros((n_images, w_in, w_in, 3)).astype(np.uint8)
        img = np.asarray(np.argmax(gt_all,axis=1))

        x[img==0] = [0.0,0.0,0.0]
        x[img==1] = [255.0,255.0,0.0]
        x[img==2] = [0.0,255.0,255.0]
        x[img==3] = [128.0,0.0,128.0]
        save_image(x, "gt")

        in_all = np.transpose(in_all, (0,2,3,1))
        x = np.asarray(in_all*255.0, dtype=np.uint8)
        save_image(x, "in")

    return make_image
