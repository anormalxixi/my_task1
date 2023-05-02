#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class myTextcnn(tf.keras.layers.Layer):
    def __init__(self,max_words,embedding_dim,max_len,filters,window_size,hidden_dim,**kwargs):
        super(myTextcnn,self).__init__(name='myTextcnn',**kwargs)
        self.embed_layer=Embedding(max_words,output_dim=embedding_dim,input_length=max_len)
        self.conv1=Conv1D(filters,window_size[0],padding='same',activation='relu')
        self.pool=MaxPooling1D(pool_size)
        self.conv2=Conv1D(filters,window_size[1],padding='same',activation='relu')
        self.conv3=Conv1D(filters,window_size[2],padding='same',activation='relu')
        
        self.concat=Concatenate()
        
        self.flatten=Flatten()
        
        self.dense1=Dense(hidden_dim,activation='relu')
        self.dense_output=Dense(num_labels+1,activation='softmax')
        
    def call(self,inputs):
        embed=self.embed_layer(inputs)
        
        cnn1=self.conv1(embed)
        cnn1=self.pool(cnn1)
        cnn2=self.conv2(embed)
        cnn2=self.pool(cnn2)
        cnn3=self.conv3(embed)
        cnn3=self.pool(cnn3)
        
        cnn=self.concat([cnn1,cnn2,cnn3])
        
        x=self.flatten(cnn)
        x=self.dense1(x)
        return self.dense_output(x)

