# Amazon_Sentiment

Project Overview: 
  A Sentiment analysis on Amazon customer descriptions and star reviews. 
  Text anlysis of product reviews will give Amazon and other companies insight
  to how their customers like their products and how any particular company 
  can best serve their customers needs. To shed some light on classifying 
  the wants and needs of customers I created a LSTM + CNN model with assistance 
  from Google's word embedding word2vec. In an attempt to isolate the most meaningful 
  words I preprocess the data taking out english and spanish words as well as expanding 
  english contractions and stemming words.  For my model, I use an untrained embedding 
  for based off weights of the words already trained by word2vec.  I use convolution
  layer to keep the relational data of words that are groups in close proximity, and 
  of course a LSTM. I then compare my findings using my model to make predictions on
  some held out testing data.  
  
  
The Nitty Gritty (How to run the code and any requirements): 
  I import the following libraries...
  1. numpy
  2. pandas
  3. keras
  4. sklearn
  5. re
  6. nltk 
  7. gensim
  8. keras
  
  you can find Google's word2vec: https://github.com/dav/word2vec
  Data: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
  Or at: https://www.kaggle.com/bittlingmayer/amazonreviews

My name is Kelly Harner and you can view a description of my code: https://youtu.be/OSpjcDvIzt4
