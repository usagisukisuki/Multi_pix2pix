#coding: utf-8
#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
iteration=1
# dc0ec6 dc1ec5

# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02) #正規分布で配列の初期化
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    iteration=1
    def __init__(self, out_ch):
        global iteration
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c3_0'] = CBR(1536, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4_0'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5_0'] = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(320, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6_0'] = CBR(256, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)
    def __call__(self, hs):
        if iteration<2:
            h = self.c0(hs[-1])
            for i in range(1,8):
                if i<4:
                    h = F.concat([h, hs[-i-1]])
                    h = self['c%d'%i](h)
                elif i>3 and i<7:
                    h = F.concat([h, hs[-i-1]])
                    h = self['c%d_0'%i](h)
                elif i>7 and i<7:
                    h = F.concat([h, hs[-i-1]])
                    h = self['c%d'%i](h)
                else:
                    h = F.concat([h, hs[-i-1]])
                    h = self.c7(h)
        return h


class Discriminator_vgg16(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(128, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = L.Convolution2D(128, 128, 3, 1, 1, initialW=w)
        layers['c3'] = L.BatchNormalization(128) 
        layers['c4'] = L.Convolution2D(128, 128, 3, 1, 1, initialW=w)
        layers['c5'] = L.BatchNormalization(128)
        layers['c6'] = L.Convolution2D(128, 256, 3, 1, 1, initialW=w)
        layers['c7'] = L.BatchNormalization(256)
        layers['c8'] = CBR(256, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c9'] = L.Convolution2D(256, 256, 3, 1, 1, initialW=w)
        layers['c10'] = L.BatchNormalization(256)
        layers['c11'] = L.Convolution2D(256, 256, 3, 1, 1, initialW=w)
        layers['c12'] = L.BatchNormalization(256)
        layers['c13'] = L.Convolution2D(256, 512, 3, 1, 1, initialW=w)
        layers['c14'] = L.BatchNormalization(512)
        layers['c15'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c16'] = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
        layers['c17'] = L.BatchNormalization(512)
        layers['c18'] = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
        layers['c19'] = L.BatchNormalization(512)
        layers['c20'] = L.Convolution2D(512, 256, 3, 1, 1, initialW=w)
        layers['c21'] = L.BatchNormalization(256)
        layers['c22'] = L.Convolution2D(256, 128, 3, 1, 1, initialW=w)
        layers['c23'] = L.BatchNormalization(128)
        layers['c24'] = L.Convolution2D(128, 1, 3, 1, 1, initialW=w)
        super(Discriminator_vgg16, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = F.leaky_relu(h)
        h = self.c4(h)
        h = self.c5(h)
        h = F.leaky_relu(h)
        h = self.c6(h)
        h = self.c7(h)
        h = F.leaky_relu(h)
        h = self.c8(h)
        h = F.leaky_relu(h)
        h = self.c9(h)
        h = self.c10(h)
        h = F.leaky_relu(h)
        h = self.c11(h)
        h = self.c12(h)
        h = F.leaky_relu(h)
        h = self.c13(h)
        h = self.c14(h)
        h = F.leaky_relu(h)
        h = self.c15(h)
        h = self.c16(h)
        h = self.c17(h)
        h = F.leaky_relu(h)
        h = self.c18(h)
        h = self.c19(h)
        h = F.leaky_relu(h)
        h = self.c20(h)
        h = self.c21(h)
        h = F.leaky_relu(h)
        h = self.c22(h)
        h = self.c23(h)
        h = F.leaky_relu(h)
        h = self.c24(h)
        #print(h.shape)
        h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h


class Discriminator_resnet(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(64, 64, bn=True, sample='up', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(64, 64, bn=True, sample='up', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c8'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c9'] = CBR(64, 64, bn=True, sample='up', activation=F.leaky_relu, dropout=False)
        layers['c10'] = CBR(64, 64, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c11'] = CBR(64, 64, bn=True, sample='up', activation=F.leaky_relu, dropout=False)
        layers['c12'] = L.Convolution2D(64, 1, 3, 1, 1, initialW=w)
        super(Discriminator_resnet, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
	#print(h.shape) #128
        h1 = self.c1(h)
	#print(h1.shape) #64
        h = self.c2(h1)
	#print(h.shape) #32
        h = self.c3(h)
	#print(h.shape) #64
        h2 = self.c4(h+h1)
	#print(h2.shape) #32
        h = self.c5(h2)
	#print(h.shape) #16
        h = self.c6(h)
	#print(h.shape) #32
        h3 = self.c7(h+h2)
	#print(h3.shape) #16
        h = self.c8(h3)
	#print(h.shape) #8
        h = self.c9(h)
	#print(h.shape) #16
        h = self.c10(h+h3)
	#print(h.shape) #16
        h = self.c11(h)
	#print(h.shape) #32
        h = self.c12(h)
        #print(h.shape)
        h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h

class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False) #64
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False) #32
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False) #16
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w) #32
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        #print(h.shape)
        return h
