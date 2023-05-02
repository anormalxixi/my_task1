#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import jieba
import json
from keras.models import model_from_json
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


from tensorflow.keras import Model


# In[19]:


#加载模型架构
with open('my_model_architecture.json')as f1:
    data=json.load(f1)
    model=model_from_json(data)


# In[20]:


#加载权重
model.load_weights('4_28model_weights.h5')


# In[5]:


#加载字典
with open ('tokenizer.json')as f2:
    data2=json.load(f2)
    tokenizer=tokenizer_from_json(data2)


# In[6]:


#test分词
stop_list=pd.read_csv('中文停用词.txt',encoding='utf-8',sep='\t',names=['stop_list'],quoting=3)
def text_cut(sentence):
    sen_cut=jieba.lcut(sentence)
    sen_cut=[w for w in sen_cut if (w not in stop_list.values) and (w >=u'\u4e00' and w<= u'\u9fa5')]
    return " ".join(sen_cut)#用空格隔开


# In[22]:


test=pd.read_csv('news_test_no_answer.csv',encoding='gb18030',index_col=0)
test['cutlist']=test['新闻'].apply(text_cut)
to_pred=test['cutlist']
to_pred=np.array(to_pred)
to_pred_seq=tokenizer.texts_to_sequences(to_pred)
to_pred_pad=pad_sequences(to_pred_seq,maxlen=200,padding='post',truncating='post')


# In[23]:


pred=model.predict(to_pred_pad)
pred=tf.argmax(pred,axis=1)
final=pred.numpy()


# In[24]:


final


# In[25]:


submission_429_2=pd.DataFrame(pd.Series(range(len(final)),name='id'))
submission_429_2['label']=pd.Series(final)   
submission_429_2.to_csv('submission_429_2.csv',index=False)
submission_429_2[:20]


# In[ ]:




