import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K

class PD(Optimizer):
    def __init__(self, lr=0.2, kd=0.1, **kwargs):
        super(PD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.kd = K.variable(kd, name='kd')

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        
        shapes = [K.int_shape(p) for p in params]
        prev_grads = [K.zeros(shape, name='prev_grad_' + str(i))
                for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + prev_grads
        
        for p, g, pg in zip(params, grads, prev_grads):
            new_p = p - self.lr * g + self.kd * (g - pg)
            self.updates.append(K.update(pg, g))
            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                'Kd': float(K.get_value(self.kd))}
        base_config = super(PD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


