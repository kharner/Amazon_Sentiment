
# coding: utf-8

# # Amazon Sentiment Data Analysis

# Datasets from Kaggle.com courtesy of Adam Mathias Bittlingmayer. The webpage can be reached through https://www.kaggle.com/bittlingmayer/amazonreviews OR https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

# ## Loading the data

# In[1]:


import numpy as np
import pandas as pd
from keras.utils import to_categorical

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score

from sklearn.metrics import confusion_matrix


# In[2]:


columns = [ 'Label', 'Title', 'Description']
rawTrainSentiments = pd.read_csv('amazon_review_full_csv/train.csv', names=columns)
full_set = rawTrainSentiments.assign(X_train = lambda x: x.Title +' '+ x.Description)


# In[3]:


rawTestSentiments = pd.read_csv('amazon_review_full_csv/test.csv', names=columns)
full_set_test = rawTestSentiments.assign(X_test = lambda x: x.Title + ' ' + x.Description)


# In[4]:


full_set['Label'] = full_set['Label'].astype(int)
full_set['Title'] = full_set['Title'].astype(str)
full_set['Description'] = full_set['Description'].astype(str)
full_set['X_train'] = full_set['X_train'].astype(str)

full_set_test['Label'] = full_set_test['Label'].astype(int) 
full_set_test['Title'] = full_set_test['Title'].astype(str)
full_set_test['Description'] = full_set_test['Description'].astype(str)

full_set_test['X_test'] = full_set_test['X_test'].astype(str)

full_set.head()


# In[5]:


y_train = np.array(full_set['Label'])
X_train = np.array(full_set['X_train'])

y_test = np.array(full_set_test['Label'])
X_test = np.array(full_set_test['X_test'])

encoded_labels = to_categorical(y_train)


# ## Preprocessing 
# Below I define a contractions dictionary to expand the english contractions.  It is necessary because I will define english, stop words and replace redundant words like 'not'.  The Amazon Sentiment data also includes reviews writen in spanish which is why I have also included Spanish stop words. Finally the word stemmer converts all english verbs into a like tense.  

# In[6]:


contractions_dictionary = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


# In[7]:


import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

english_stop_words = set(stopwords.words('english'))
spanish_stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('english')


# In[8]:


def preprocess_data(text):
    text = text.lower()
    regEx = re.compile('(%s)' % '|'.join(contractions_dictionary.keys()))

    def expand_contractions(s, contractions_dictionary=contractions_dictionary):
        def replace(match):
            return contractions_dictionary[match.group(0)]
        return regEx.sub(replace, s)
    
    text = expand_contractions(text, contractions_dictionary)
    text = text.split()
        
    text = [ew for ew in text if not ew in english_stop_words]
    text = [sw for sw in text if not sw in spanish_stop_words]
    
    stem = [stemmer.stem(w) for w in text]
    text = " ".join(stem)
    
    return text


# In[9]:


X_train = [preprocess_data(text) for text in full_set['X_train']]
X_test = [preprocess_data(text) for text in full_set_test['X_test']]


# ## Transfer Learning
# I utilized the power of Google's neural network Word2Vec, trained and learned word embeddings as the basis of weights in my own model.

# In[10]:


from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[11]:


from keras.preprocessing.text import Tokenizer
MAX_VECTOR_WORDS = 100000

tokenizer = Tokenizer(num_words=MAX_VECTOR_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', split=' ', char_level=False, oov_token=None)
tokenizer.fit_on_texts(X_train + X_test)


# In[14]:


from keras.preprocessing import sequence
vocab_size = len(tokenizer.word_index)

kMAXLEN = 300     #want all comment descriptions to be of size 300 when training
kVECTORLEN = 300  #the size of each vector 


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


X_train_data = sequence.pad_sequences(X_train_seq, maxlen=kMAXLEN)
X_test_data = sequence.pad_sequences(X_test_seq, maxlen=kMAXLEN)


# In[15]:


nb_words = len(tokenizer.word_index)
embedded_matrix = np.zeros((nb_words, kVECTORLEN))

for word, i in tokenizer.word_index.items():
    if word in word2vec.vocab and i < nb_words:
        embedded_matrix[i] = word2vec.word_vec(word)


# ## THE MODEL
# After mapping words to corresponding digits and filling the embedded matrix its time to finally train the model.  
# I use an embedded matrix for the first layer where the weights are borrowed from the genius of Google's Word2Vec.
# Then I use an convolutional layer to relate words near each other. I also sandwich my LSTM model with dropout layers since they are prone to overfitting.  

# In[16]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

vocab_size = len(tokenizer.word_index)
kTOP = vocab_size
model = Sequential()
model.add( Embedding(kTOP,kVECTORLEN, input_length=kMAXLEN, weights=[embedded_matrix], trainable=False) )
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.1))
model.add(LSTM(150))
model.add(Dropout(0.1))
model.add(Dense(6, activation='relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# In[18]:


history = model.fit(X_train_data, y=encoded_labels, batch_size=32, nb_epoch=1, verbose=1, validation_split=0.2, shuffle=True)


# ## Analysis 
# 

# In[19]:


print('The model accuracy is: ', history.history['acc'])


# In[21]:


test_predictions = model.predict(X_test_data)
test_class_predictions = test_predictions.argmax(axis=-1)


# In[22]:


print('F1 score: ', f1_score(y_test,test_class_predictions,average='micro'))


# In[23]:


print('Recall score: ', recall_score(y_test,test_class_predictions,average='micro'))


# In[24]:


print('Precision: ', precision_score(y_test,test_class_predictions,average='weighted'))


# In[25]:


print('Confusion Matrix: \n', confusion_matrix(y_test, test_class_predictions) )

