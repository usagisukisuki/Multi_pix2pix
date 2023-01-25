#coding: utf-8
#!/usr/bin/env python

import numpy as np
from sklearn.metrics import confusion_matrix
import chainer
import chainer.cuda
from chainer import Variable

def IoU_val(updater, enc, dec, rows, cols):
    @chainer.training.make_extension()
    def make_IoU_val(trainer):
        def semantic_segmentation_confusion(pred_labels, truth_labels):
            n_class = 0
            confusion = np.zeros((n_class, n_class), dtype=np.int64)
            pred_label = pred_labels.flatten()
            truth_label = truth_labels.flatten()

            lb_max = np.max((pred_label, truth_label))
            if lb_max >= n_class:
                expanded_confusion = np.zeros((lb_max + 1, lb_max + 1))
                expanded_confusion[0:n_class, 0:n_class] = confusion

                n_class = lb_max + 1
                confusion = expanded_confusion
                confusion += np.bincount(n_class * truth_label.astype(np.int32) + pred_label, minlength=n_class**2).reshape((n_class, n_class))

                return confusion

        def semantic_segmentation_iou(confusion):
            iou_den = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
            iou = np.array(np.diag(confusion) ,dtype = np.float32) / np.array(iou_den, dtype = np.float32)
            return iou

        def eval_semantic_segmentation(pred_labels, truth_labels):
            confusion = semantic_segmentation_confusion(pred_labels, truth_labels)
            iou = semantic_segmentation_iou(confusion)
            pixel_accuracy = np.array(np.diag(confusion).sum(),dtype = np.float32) / np.array(confusion.sum(),dtype = np.float32)
            class_accuracy = np.array(np.diag(confusion), dtype = np.float32) / np.array(np.sum(confusion, axis=1), dtype = np.float32)

            haikei = iou[0]
            membranes = iou[1]
            mitochondria = iou[2]
            synapses = iou[3]
            iou = np.nanmean(iou)

            return haikei,membranes,mitochondria,synapses,iou

        n_images = rows * cols
        xp = enc.xp

        w_in = 256
        w_out = 256
        in_ch = 1
        out_ch = 4 #nomber of classes

        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")

        haikei_all = 0
        kaku_all = 0
        iou_all = 0
        for it in range(n_images):
            batch = updater.get_iterator('val').next()
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
                    z = enc(x_in)
                    x_out = dec(z)

            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.data.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]
            #print(gen_all.shape)
            y_ans = np.argmax(gen_all,axis=1).astype(np.int32)
            y_truth = np.argmax(gt_all,axis=1).astype(np.int32)
	    #y_label = chainer.cuda.to_cpu(np.argmax(y.data[i] ,axis=0).astype(np.int32))
	    #print(y_ans.shape)
	    #print(y_truth)


        haikei,membranes,mitochondria,synapses,iou = eval_semantic_segmentation(y_ans,y_truth)

	#print(iou)
        #print(maku)
        #print(kaku)
        #haikei = haikei / n_images
        #kaku = kaku / n_images
        #iou = iou / n_images
        chainer.report({'haikei_val': haikei})
        chainer.report({'membranes_val': membranes})
        chainer.report({'mitochondria_val': mitochondria})
        chainer.report({'synapses_val': synapses})
        chainer.report({'IoU_val': iou})


    return make_IoU_val
