"""
    PID controller based optimizer

    This method proposed by W. An et al.
    
    # References
        - [A PID Controller Approach for Stochastic Optimization of Deep Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/An_A_PID_Controller_CVPR_2018_paper.pdf)
"""

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K

class PID(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, kd=0.1, **kwargs):
        super(PID, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.kd = K.variable(kd, name='kd')

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        
        shapes = [K.int_shape(p) for p in params]
        prev_grads = [K.zeros(shape, name='prev_grad_' + str(i))
                for (i, shape) in enumerate(shapes)]
        ds = [K.zeros(shape, name='d_' + str(i)) 
                for (i, shape) in enumerate(shapes)]
        vs = [K.zeros(shape, name='v_' + str(i)) 
                for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + ds + vs + prev_grads
        
        for p, g, pg, v, d in zip(params, grads, prev_grads, vs, ds):
            v_t = self.momentum * v - self.lr * g
            self.updates.append(K.update(v, v_t))

            d_t = self.momentum * d + (1 - self.momentum) * (g - pg)
            self.updates.append(K.update(d, d_t))
            self.updates.append(K.update(pg, g))
            
            new_p = p + v_t + self.kd * d_t
            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                'momentum': float(K.get_value(self.momentum)),
                'Kd': float(K.get_value(self.kd))}
        base_config = super(PID, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


