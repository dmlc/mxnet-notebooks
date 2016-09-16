import numpy as np
import mxnet as mx

class SimpleBatch(object):
    def __init__(self, data, label, pad=None):
        self.data = data
        self.label = label
        self.pad = pad

class SimpleIter:
    def __init__(self, mu, sigma, batch_size, num_batches):
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data_shape = (batch_size, mu.shape[1])
        self.label_shape = (batch_size, )
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data', self.data_shape)]

    @property
    def provide_label(self):
        return [('softmax_label', self.label_shape)]

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            num_classes = self.mu.shape[0]
            label = np.random.randint(0, num_classes, self.label_shape)
            data = np.zeros(self.data_shape)
            for i in range(num_classes):
                data[label==i,:] = np.random.normal(
                    self.mu[i,:], self.sigma[i,:], (sum(label==i), self.data_shape[1])) 
            return SimpleBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)], pad=0)
        else:
            raise StopIteration

class SyntheticData:
    """Genrate synthetic data
    """
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.mu = np.random.rand(num_classes, num_features)
        self.sigma = np.ones((num_classes, num_features)) * 0.1

    def get_iter(self, batch_size, num_batches=10):
        return SimpleIter(self.mu, self.sigma, batch_size, num_batches)