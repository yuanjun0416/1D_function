
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

class simam_module(layers.Layer):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activation = layers.Activation('sigmoid')
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__+'('
        s = s + ('lambda=%f' % self.e_lambda) + ')'

        return s
    
    @staticmethod
    def get_module_name():
        return "simam"
    
    def call(self, x):
        batch, time_steps, channel = x.shape

        n = time_steps - 1

        x_minus_mu_square = tf.square(x - tf.reduce_mean(x, axis=1, keepdims=True))
        y = x_minus_mu_square/(4*(tf.reduce_sum(x_minus_mu_square, axis=1, keepdims=True)/n + self.e_lambda)) + 0.5
        # print(y)
        
        return x * self.activation(y)

# simam = simam_module()
# x = keras.initializers.RandomNormal()(shape=(1, 81, 2))
# # x = tf.convert_to_tensor(x)
# print(x)
# sim = simam(x)

# print(sim)