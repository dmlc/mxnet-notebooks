# pylint: skip-file
import mxnet as mx
import numpy as np
import logging
import model
from BCD_one import BCD_one

class Monitor(object):
    def __init__(self, interval, level=logging.DEBUG, stat=None):
        self.interval = interval
        self.level = level
        if stat is None:
            def mean_abs(x):
                return np.fabs(x).mean()
            self.stat = mean_abs
        else:
            self.stat = stat

    def forward_end(self, i, internals):
        if i%self.interval == 0 and logging.getLogger().isEnabledFor(self.level):
            for key in sorted(internals.keys()):
                arr = internals[key]
                logging.log(self.level, 'Iter:%d  param:%s\t\tstat(%s):%s'%(i, key, self.stat.__name__, str(self.stat(arr.asnumpy()))))

    def backward_end(self, i, weights, grads, metric=None):
        if i%self.interval == 0 and logging.getLogger().isEnabledFor(self.level):
            for key in sorted(grads.keys()):
                arr = grads[key]
                logging.log(self.level, 'Iter:%d  param:%s\t\tstat(%s):%s\t\tgrad_stat:%s'%(i, key, self.stat.__name__, str(self.stat(weights[key].asnumpy())), str(self.stat(arr.asnumpy()))))
        if i%self.interval == 0 and metric is not None:
                logging.log(logging.INFO, 'Iter:%d metric:%f'%(i, metric.get()[1]))
                metric.reset()

class Solver(object):
    def __init__(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            self.optimizer = mx.optimizer.create(optimizer, **kwargs)
        else:
            self.optimizer = optimizer
        self.updater = mx.optimizer.get_updater(self.optimizer)
        self.monitor = None
        self.metric = None
        self.iter_end_callback = None
        self.iter_start_callback = None

    def set_metric(self, metric):
        self.metric = metric

    def set_monitor(self, monitor):
        self.monitor = monitor

    def set_iter_end_callback(self, callback):
        self.iter_end_callback = callback

    def set_iter_start_callback(self, callback):
        self.iter_start_callback = callback

    def solve(self, X, R, V, lambda_v_rt, lambda_u, lambda_v, dir_save, batch_size, xpu, sym, args, args_grad, auxs,
              data_iter, begin_iter, end_iter, args_lrmult={}, debug = False):
        # names and shapes
        input_desc = data_iter.provide_data + data_iter.provide_label
        input_names = [k for k, shape in input_desc]
        # plances to store them
        input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in input_desc]
        args = dict(args, **dict(zip(input_names, input_buffs)))

        # list all outputs (strings)
        output_names = sym.list_outputs()
        if debug:
            sym = sym.get_internals()
            blob_names = sym.list_outputs()
            sym_group = []
            for i in range(len(blob_names)):
                if blob_names[i] not in args:
                    x = sym[i]
                    if blob_names[i] not in output_names:
                        x = mx.symbol.BlockGrad(x, name=blob_names[i])
                    sym_group.append(x)
            sym = mx.symbol.Group(sym_group)
        # bind the network params to the network (symbol)
        exe = sym.bind(xpu, args=args, args_grad=args_grad, aux_states=auxs)

        assert len(sym.list_arguments()) == len(exe.grad_arrays)
        update_dict = {name: nd for name, nd in zip(sym.list_arguments(), exe.grad_arrays) if nd}
        batch_size = input_buffs[0].shape[0]
        self.optimizer.rescale_grad = 1.0/batch_size
        self.optimizer.set_lr_mult(args_lrmult)

        output_dict = {}
        output_buff = {}
        internal_dict = dict(zip(input_names, input_buffs))
        # exe.outputs is a list of all output ndarrays
        for key, arr in zip(sym.list_outputs(), exe.outputs):
            if key in output_names:
                output_dict[key] = arr
                output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
            else:
                internal_dict[key] = arr

        # init' U
        U = np.mat(np.zeros((R.shape[0],V.shape[1])))
        # set lambda_v_rt to 0 in the first epoch
        lambda_v_rt_old = np.zeros(lambda_v_rt.shape)
        lambda_v_rt_old[:] = lambda_v_rt[:]
        lambda_v_rt[:,:] = 0
        epoch = 0 # index epochs
        data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            lambda_v_rt},
            batch_size=batch_size, shuffle=False,
            last_batch_handle='pad')
        data_iter.reset()
        for i in range(begin_iter, end_iter):
            if self.iter_start_callback is not None:
                if self.iter_start_callback(i):
                    return
            #if i==100:
            #    V = np.zeros(V.shape)
            #    data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
            #        lambda_v_rt},
            #        batch_size=batch_size, shuffle=False,
            #        last_batch_handle='pad')
            #    data_iter.reset()
            #    for j in range(10):
            #        batch = data_iter.next()
            try:
                batch = data_iter.next()
            except:
                # means the end of an epoch
                epoch += 1
                theta = model.extract_feature(sym[0], args, auxs,
                    data_iter, X.shape[0], xpu).values()[0]
                # update U, V and get BCD loss
                U, V, BCD_loss = BCD_one(R, U, V, theta,
                    lambda_u, lambda_v, dir_save, True)
                # get recon' loss
                Y = model.extract_feature(sym[1], args, auxs,
                    data_iter, X.shape[0], xpu).values()[0]
                Recon_loss = lambda_v/np.square(lambda_v_rt_old[0,0])*np.sum(np.square(Y-X))/2.0
                print "Epoch %d - tr_err/bcd_err/rec_err: %.1f/%.1f/%.1f" % (epoch,
                    BCD_loss+Recon_loss, BCD_loss, Recon_loss)
                fp = open(dir_save+'/cdl.log','a')
                fp.write("Epoch %d - tr_err/bcd_err/rec_err: %.1f/%.1f/%.1f\n" % (epoch,
                    BCD_loss+Recon_loss, BCD_loss, Recon_loss))
                fp.close()
                lambda_v_rt[:] = lambda_v_rt_old[:] # back to normal lambda_v_rt
                data_iter = mx.io.NDArrayIter({'data': X, 'V': V, 'lambda_v_rt':
                    lambda_v_rt},
                    batch_size=batch_size, shuffle=False,
                    last_batch_handle='pad')
                data_iter.reset()
                batch = data_iter.next()

            for data, buff in zip(batch.data+batch.label, input_buffs):
                # copy data from batch to input_buffs
                # input_buffs is used during ff and bp
                # buffs->args->exe
                data.copyto(buff)
            exe.forward(is_train=True)
            if self.monitor is not None:
                self.monitor.forward_end(i, internal_dict)
            for key in output_dict:
                # output_buff is used for computing metrics
                output_dict[key].copyto(output_buff[key])

            exe.backward()
            for key, arr in update_dict.items():
                self.updater(key, arr, args[key])

            if self.metric is not None:
                self.metric.update([input_buffs[-1]],
                                   [output_buff[output_names[0]]])

            if self.monitor is not None:
                self.monitor.backward_end(i, args, update_dict, self.metric)

            if self.iter_end_callback is not None:
                if self.iter_end_callback(i):
                    return
            exe.outputs[0].wait_to_read()
        #Y = model.extract_feature(sym[0], args, auxs,
        #        data_iter, X.shape[0], xpu).values()[0]
        #print Y
        #print Y.shape
        theta = model.extract_feature(sym[0], args, auxs,
            data_iter, X.shape[0], xpu).values()[0]
        U, V, BCD_loss = BCD_one(R, U, V, theta, lambda_u, lambda_v,
            dir_save, True, 1)
        fp.close()
        return U, V, theta, BCD_loss
