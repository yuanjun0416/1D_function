from tensorflow.keras import layers, Sequential

class spp(layers.Layer):
    def __init__(self):
        super(spp, self).__init__()

        self.maxpool_5 = layers.MaxPool1D(pool_size=5, strides=1, padding='same')
        self.maxpool_9 = layers.MaxPool1D(pool_size=9, strides=1, padding='same')
        self.maxpool_13 = layers.MaxPool1D(pool_size=13, strides=1, padding='same')

    def call(self, x):
        spp_x = x
        x5 = self.maxpool_5(x)
        x9 = self.maxpool_9(x)
        x13 = self.maxpool_13(x)
        y = layers.Concatenate(axis=-1)([spp_x, x5, x9, x13])
        
        return y
    
class sppf(layers.Layer):
    def __init__(self):
        super(sppf, self).__init__()

        self.conv1 = Sequential([
                               layers.Conv1D(128, kernel_size=1, strides=1, padding='same'),
                               layers.BatchNormalization(),
                               layers.Activation('relu')])
        self.maxpool = layers.MaxPool1D(pool_size=5, strides=1, padding='same')

    def call(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)

        x = layers.Concatenate(axis=-1)([x, x1, x2, x3])

        return x