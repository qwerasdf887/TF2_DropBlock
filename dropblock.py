import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


'''
tensorflow drop layer in training phase:
output = input / survival prob
survival prob = 1 - drop rate
i.e., 全1的輸入進入drop layer(假設drop rate = 0.4)
則輸出無丟棄的node值會是 1 / ( 1 - 0.4)
EfficientNet 的 drop_connect也是用此方式決定輸出值

但是DropBlock論文中，寫 A * count(M) / count_ones(M) 來當作輸出值
count(M): M的元素個數
count_ones(M): M中為1的個數

不確定是否採用算好的M來計算，但根據Tensorflow中的Drop與EfficientNet推斷，
採用第一種方式。
i.e., A = A / survival prob
'''

class DropBlock(tf.keras.layers.Layer):
    #drop機率、block size
    def __init__(self, drop_rate=0.2, block_size=3, **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.rate = drop_rate
        self.block_size = block_size

    def call(self, inputs, training=None):
        if training:
            '''
            feature map mask tensor
            創建一個均勻取樣的Tensor，加上drop rate之後取整數，則為1的部份表示drop block的中心點
            經由maxpooling將中心點擴散成block size大小
            最後將Tensor值更改(0->1, 1->0)
            則可形成mask
            '''
            #batch size
            b = tf.shape(inputs)[0]
            
            random_tensor = tf.random.uniform(shape=[b, self.m_h, self.m_w, self.c]) + self.bernoulli_rate
            binary_tensor = tf.floor(random_tensor)
            binary_tensor = tf.pad(binary_tensor, [[0,0],
                                                   [self.m_h // 2, self.m_h // 2],
                                                   [self.m_w // 2, self.m_h // 2],
                                                   [0, 0]])
            binary_tensor = tf.nn.max_pool(binary_tensor,
                                           [1, self.block_size, self.block_size, 1],
                                           [1, 1, 1, 1],
                                           'SAME')
            binary_tensor = 1 - binary_tensor
            inputs = tf.math.divide(inputs, (1 - self.rate)) * binary_tensor
        return inputs
    
    def get_config(self):
        config = super(DropBlockConv2D, self).get_config()
        return config

    def build(self, input_shape):
        #feature map size (height, weight, channel)
        self.b, self.h, self.w, self.c = input_shape.as_list()
        #mask h, w
        self.m_h = self.h - (self.block_size // 2) * 2
        self.m_w = self.w - (self.block_size // 2) * 2
        self.bernoulli_rate = (self.rate * self.h * self.w) / (self.m_h * self.m_w * self.block_size**2)



if __name__ == "__main__":
    import numpy as np
    inputs = tf.keras.Input(shape=(5, 5, 3))
    x = DropBlock()(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.summary()

    test = np.ones([2,5,5,3], dtype=np.float32)

    print('in test phase:\n')
    print(model.predict(test))
    print('----------------------------------------------')
    print('in training phase:\n')
    print(model(test, training=True))