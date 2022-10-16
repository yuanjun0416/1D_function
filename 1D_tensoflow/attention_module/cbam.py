import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

class ChannelAttention(layers.Layer):
    def __init__(self, channel, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

        # self.inputs = inputs
        # channel = self.inputs.shape[-1]
        
        self.channel = channel  #输入数据的最后一维, 即输入通道
        self.avg_pool = layers.GlobalAveragePooling1D()  

        self.max_pool = layers.GlobalMaxPool1D()

        self.share_layer_one = layers.Dense(channel//ratio, 
                                            activation='relu',
                                            kernel_initializer='he_normal',
                                            use_bias=True,
                                            bias_initializer='zeros')

        self.share_layer_two = layers.Dense(channel,
                                            kernel_initializer='he_normal',
                                            use_bias=True,
                                            bias_initializer='zeros')
        
        self.add = layers.Add()
        self.act = layers.Activation('sigmoid')
    
    def call(self, inputs):
        self.inputs = inputs

        avg_pool = self.avg_pool(self.inputs)   # input: [N, time_step, C] output: [N, C]
        avg_pool = layers.Reshape((1, self.channel))(avg_pool) # input: [N, C] output: [N, 1, C]

        max_pool = self.max_pool(self.inputs)   # input: [N, time_step, C] output: [N, C]
        max_pool = layers.Reshape((1, self.channel))(max_pool) # input: [N, C] output: [N, 1, C]

        avg_pool = self.share_layer_one(avg_pool) # input [N, 1, C] output: [N, 1, C/ratio]
        avg_pool = self.share_layer_two(avg_pool) # input [N, 1, C/ratio] output: [N, 1, C]
        max_pool = self.share_layer_one(max_pool) # input [N, 1, C] output: [N, 1, C/ratio]
        max_pool = self.share_layer_two(max_pool) # input [N, 1, C/ratio] output: [N, 1, C]
        
        cbam_feature = self.add([avg_pool, max_pool]) # input: [N, 1, C] ouput: [N, 1, C]
        cbam_feature = self.act(cbam_feature)      
        # print(cbam_feature.shape)

        ## return 广播机制: [N, 1, C]*[N, time_step, C]=[N, time_step, C]
        return layers.multiply([self.inputs, cbam_feature])  ## 等同于*, 广播机制: 两个数组的后缘维度相同, 或者在其中一方的维度为1。广播在缺失或者长度为1的维度上进行补充
        

class SpatialAttention(layers.Layer):
    def __init__(self,  kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

        self.kernel_size = kernel_size # 卷积核的大小
        # self.inputs = inputs

        self.concatenate = layers.Concatenate(axis=2) # 在最后一维, 即通道进行拼接

        self.conv = layers.Conv1D(filters=1,
                                  kernel_size=self.kernel_size,
                                  strides=1,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False)
    def call(self, inputs):
        self.inputs = inputs

        avg_pool = keras.backend.mean(self.inputs, axis=2, keepdims=True) # input: [N, time_step, C] ouput: [N, time_step, 1]
        max_pool = keras.backend.max(self.inputs, axis=2, keepdims=True)  # input: [N, time_step, C] ouput: [N, time_step, 1]

        concat = self.concatenate([max_pool, avg_pool]) # ouput: [N, time_step, 2]

        cbam_feature = self.conv(concat)     # input: [N, time_step, 2] ouput: [N, time_step, 1]

        assert cbam_feature.shape[-1] == 1   


        # return 广播机制: [N, time_step, 1] * [N, time_step, C] = [N, time_step, C]
        return layers.multiply([self.inputs, cbam_feature])
        

