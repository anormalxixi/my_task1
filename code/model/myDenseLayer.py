#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class myDenseLayer(tf.keras.layers.Layer):
    def __init__(self,num_output,activation=None,kernel_regularizer=None,**kwargs):
        super(myDenseLayer,self).__init__(**kwargs)
        self.num_output=num_output
        self.activation=activations.get(activation)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        
    def build(self,input_shape):
        self.kernel=self.add_weight("kernel",
                                   shape=(int(input_shape[-1]),
                                   self.num_output),
                                   regularizer=self.kernel_regularizer,
                                   initializer='random_normal',
                                   trainable=True)
        self.bias=self.add_weight("bias",
                                 shape=(self.num_output,),
                                 initializer='zeros',
                                 trainable=True)
        super(myDenseLayer,self).build(input_shape)
    def call(self,inputs):
        return self.activation(tf.matmul(inputs,self.kernel)+self.bias)

