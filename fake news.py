# %%
!pip install --upgrade tensorflow

# %%
!pip install plotly
!pip install --upgrade nbformat
!pip install nltk
!pip install spacy 
# spaCy is an open-source software library for advanced natural language processing
!pip install WordCloud
!pip install gensim 
# Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
# setting the style of the notebook to be monokai theme
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them.


# %%


# %%
!pip install jupyterthemes

# %%
!pip install plotly
!pip install --upgrade nbformat
!pip install nltk
!pip install spacy 
# spaCy is an open-source software library for advanced natural language processing
!pip install WordCloud
!pip install gensim 
# Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
# setting the style of the notebook to be monokai theme
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them.


# %%
df_true=pd.read_csv("True.csv")
df_fake=pd.read_csv("Fake.csv")

# %%
df_true

# %%
df_true.isnull().sum()

# %%
df_fake.isnull().sum()

# %%
df_fake.info()

# %%
df_true.info()

# %%
df_true['isfake']=0
df_true.head()

# %%
df_fake['isfake']=1
df_fake.head()

# %%
df=pd.concat([df_true,df_fake]).reset_index(drop=True)
df

# %%
df.drop(columns=['date'],inplace=True)

# %%
df['original']=df['title']+' '+df['text']
df.head()

# %%
df['original'][0]

# %%
nltk.download("stopwords")

# %%
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

# %%
stop_words

# %%
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len (token)>3 and token not in stop_words:
            result.append(token)
    return result

# %%
df['clean']=df['original'].apply(preprocess)

# %%
df

# %%
df['original'][0]

# %%
print(df['clean'][0])

# %%
df

# %%
list_of_words=[]
for i in df.clean:
    for j in i:
        list_of_words.append(j)

# %%
list_of_words

# %%
len(list_of_words)

# %%
total_words=len(list(set(list_of_words)))
total_words

# %%
df['clean_joined']=df['clean'].apply(lambda x: " ".join(x))

# %%
df

# %%
df['clean_joined'][0]

# %%
df['clean_joined'][0]

# %%
df['original'][1]

# %%
print(df['clean'][1])

# %%
df['clean_joined'][1]

# %%
plt.figure(figsize=(8,8))
sns.countplot(y="subject",data=df)

# %%
plt.figure(figsize=(8,8))
sns.countplot(y="isfake",data=df)

# %%
# most common words in fake news
plt.figure(figsize=(20,20))
wc=WordCloud(max_words=2000,width=1600,height=800,stopwords=stop_words).generate(" ".join(df[df.isfake==1].clean_joined))                               
plt.imshow(wc, interpolation='bilinear')

# %%
# most common words in real news
# most common words in fake news
plt.figure(figsize=(20,20))
wc=WordCloud(max_words=2000,width=1600,height=800,stopwords=stop_words).generate(" ".join(df[df.isfake==0].clean_joined))                               
plt.imshow(wc, interpolation='bilinear')



# %%
nltk.word_tokenize(df['clean_joined'][0])

# %%
maxlen=-1
for doc in df.clean_joined:
    tokens=nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen=len(tokens)
print("the max num. of words in any doc is = ",maxlen)

# %%
import plotly.express as px
fig=px.histogram(x=[len(nltk.word_tokenize(x)) for x in df.clean_joined],nbins=100)
fig.show()

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.clean_joined,df.isfake,test_size=0.2)

# %%
from nltk import word_tokenize

# %%
tokenizer=Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences=tokenizer.texts_to_sequences(x_train)
test_sequences=tokenizer.texts_to_sequences(x_train)

# %%
len(train_sequences)

# %%
train_sequences

# %%
len(test_sequences)

# %%
print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

# %%
padded_train=pad_sequences(train_sequences, maxlen=4405,padding='post',truncating='post')
padded_test=pad_sequences(test_sequences, maxlen=4405,truncating='post')

# %%
for i,doc in enumerate(padded_train[:2]):
    print("The padded encoding for document",i+1," is : ",doc)

# %%
model=Sequential()

# %%
model.add(Embedding(total_words,output_dim=128))

# %%
model.add(Bidirectional(LSTM(128)))

# %%
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.summary()

# %%
total_words

# %%
y_train=np.asarray(y_train)

# %%
model.fit(padded_train,y_train,batch_size=64,validation_split=0.1,epochs=2)

# %%
pred=model.predict(padded_test)

# %%
prediction=[]
for i in range(len(pred)):
    if pred[i].item()>0.5:
        prediction.appned(1)
    else:
        prediction.append(0)

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy=accuracy_score(list(y_test),prediction)
print("Model Accuracy : ",accuracy)

# %%
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(list(y_test),prediction)
pltfigure(figsize=(25,25))
sns.heatmap(cm, annot=True)

# %%



