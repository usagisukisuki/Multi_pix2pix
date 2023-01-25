#coding: utf-8
#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis_vgg, self.dis_res, self.dis = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)


    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        #chainer.report({'loss': loss}, enc)
        return loss

    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        #chainer.report({'loss': loss}, dec)
        return loss


    def loss_dis_vgg(self, dis_vgg, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        #chainer.report({'loss': loss}, dis_vgg)
        return loss

    def loss_dis_res(self, dis_res, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape

        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        #chainer.report({'loss': loss}, dis_res)
        return loss

    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape

        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        #chainer.report({'loss': loss}, dis)
        return loss

    #def loss_dis(self, dis, y_in, y_out):
        #batchsize,_,w,h = y_in.data.shape

        #L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        #L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        #loss = L1 + L2
        #chainer.report({'loss': loss}, dis)
        #return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_vgg_optimizer = self.get_optimizer('dis_vgg')
        dis_res_optimizer = self.get_optimizer('dis_res')
        dis_optimizer = self.get_optimizer('dis')
        #dis_optimizer = self.get_optimizer('dis')

        enc, dec, dis_vgg, dis_res, dis= self.enc, self.dec, self.dis_vgg, self.dis_res, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in = 256
        w_out = 256

        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")
        #t_out_1 = xp.zeros((batchsize, w_out, w_out)).astype(np.int32)

        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        #    t_out_1[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)

        z = enc(x_in)
        x_out = dec(z)


        y_fake_vgg = dis_vgg(x_in, x_out)
        y_fake_res = dis_res(x_in, x_out)
        y_fake = dis(x_in, x_out)

        #y_fake = dis(y_fake_1, y_fake_2, y_fake_3, test=False)

        y_real_vgg = dis_vgg(x_in, t_out)
        y_real_res = dis_res(x_in, t_out)
        y_real = dis(x_in, t_out)

        #y_real = dis(y_real_1, y_real_2, y_real_3, test=False)

        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake_vgg)
        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake_res)
        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake_vgg)
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake_res)
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_vgg_optimizer.update(self.loss_dis_vgg, dis_vgg, y_real_vgg, y_fake_vgg)
        dis_res_optimizer.update(self.loss_dis_res, dis_res, y_real_res, y_fake_res)
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)


        #dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)
