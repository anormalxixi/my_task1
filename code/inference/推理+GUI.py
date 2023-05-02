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
from tensorflow.keras.layers import Input,Embedding,Conv1D,MaxPooling1D,GlobalMaxPooling1D,Dense,Flatten
from tensorflow.keras.layers import concatenate


# In[3]:


#加载模型架构
with open('my_model_architecture.json')as f3:
    data2=json.load(f3)
    model=model_from_json(data2)


# In[4]:


#加载字典
with open ('tokenizer.json')as f2:
    data=json.load(f2)
    tokenizer=tokenizer_from_json(data)


# In[5]:


#加载权重
model.load_weights('4_28model_weights.h5')


# In[12]:


stop_list=pd.read_csv('中文停用词.txt',encoding='utf-8',sep='\t',names=['stop_list'],quoting=3)
def text_cut(sentence):
    sen_cut=jieba.lcut(sentence)
    sen_cut=[w for w in sen_cut if (w not in stop_list.values) and (w >=u'\u4e00' and w<= u'\u9fa5')]
    return " ".join(sen_cut)#用空格隔开


# In[6]:


import gradio as gr


# In[7]:


label_dict={'娱乐':1, '财经':2, '时尚':3, '房产':4,
'游戏':5, '科技':6, '家居':7, '时政':8, '教育':9, '体育':10}


# In[8]:


reverse_label_dict = dict([(value,key)for(key,value) in label_dict.items()])


# In[16]:


def process_pred_news(sen):
    sen_cut=text_cut(sen)
    list4sen=[]
    list4sen.append(sen_cut)
    sen_seq=tokenizer.texts_to_sequences(list4sen)
    sen_pad=pad_sequences(sen_seq,maxlen=200,padding='post',truncating='post')
    pred4user=model.predict(sen_pad)
    pred4user=tf.argmax(pred4user,axis=1)
    final_pred=reverse_label_dict.get(int(pred4user))
    return final_pred


# In[20]:


demo = gr.Interface(fn=process_pred_news,
                    inputs=gr.Textbox(lines=10,placeholder='News here...',label='News'),
                    outputs=gr.Textbox(label='label of the news!'),
                    title='Predict Chinese News',
                    examples=['台当局称遭索马里海盗挟持渔船船员平安中新网4月12日电 据“中央社”报道，台湾渔船“稳发161号”被索马里海盗挟持，台涉外部门发言人陈铭政11日表示，据掌握讯息显示，目前船上30名人员都很平安，船只也朝原先掌握的预定方向前进。“稳发161号”渔船6日在南纬1度51分、东经55度25分的塞席尔群岛附近海域被海盗登船挟持，台涉外部门已洽请相关组织协助处理。陈铭政接受“中央社”访问指出，台涉外部门持续掌握被挟持船只的行进方向，稳发161号仍朝西北方向前进。陈铭政强调，台涉外部门与船东保持密切联系，船东盼低调处理，以利未来谈判，充分尊重船东意愿。','哈佛北京校友会设立“哈佛中国学子基金”哈佛北京校友会日前设立“哈佛北京校友会中国学子基金”(Harvard Club of Beijing Scholarship Fund)， 以奖学金的形式资助被哈佛录取的优秀中国学子完成学业。奖学金之发放以获奖人成功就读于哈佛大学为前提。2010-2011年度奖学金的资助对象为考取哈佛大学各研究生院并将于2010年秋季入学的学子。奖学金于近日开始接受申请，具体事宜请登录哈佛北京校友会网站(http://www.beijingharvardclub.com)查询。哈佛北京校友会最早成立于1930年，现有正式会员五百多人，他们曾分别在哈佛大学各个学院或研究所/中心留学、进修或从事研究。校友会荟萃了高校和科研机构、企业、政府机构、非盈利组织的众多精英。哈佛北京校友会的宗旨：一是发扬“崇尚真知”的哈佛传统，服务于国家和社会；二是增强校友之间的联谊；三是促进校友与母校的联系、回报母校。(朱莉)'],
                    theme=gr.themes.Soft(),
                    description='It can help you classify the label of a piece of Chinese News.Have a try!',
                    
                   )


# In[21]:


demo.launch(share=True)


# In[ ]:




