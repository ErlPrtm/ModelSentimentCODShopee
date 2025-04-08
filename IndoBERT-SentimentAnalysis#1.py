# %% Import libraries
import pandas as pd
import numpy as np

review = pd.read_excel('Dataset/Data ulasan Shopee tentang COD(2).xlsx')
# print(review.head(5))

# %% Preprocessing
import pandas as pd
import string
import re
import json
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord 
from nlp_id.lemmatizer import Lemmatizer 

#import kamus bahasa baku
with open('Dataset/combined_slang_words.txt') as f:
    data0 = f.read()
print("Data type before reconstruction : ", type(data0))
formal_indo = json.loads(data0)
print("Data type after reconstruction : ", type(formal_indo), '\n')

def informal_to_formal_indo(text):
    res = " ".join(formal_indo.get(ele, ele) for ele in text.split())
    return(res)

tokenizer = Tokenizer()
stopword = StopWord()
lemmatizer = Lemmatizer()

def my_tokenizer(doc):
    doc = re.sub(r'@[A-Za-z0-9]+', '', doc)
    doc = re.sub(r'#[A-Za-z0-9]+', '', doc)
    doc = re.sub(r'RT[\s]', '', doc)
    doc = re.sub(r"http\S+", '', doc)
    doc = re.sub(r'[0-9]+', '', doc)
    doc = re.sub(r'(.)\1+',r'\1\1', doc)
    doc = re.sub(r'[\?\.\!]+(?=[\?.\!])', '',doc)
    doc = re.sub(r'[^a-zA-Z]',' ', doc)
    doc = re.sub(r'\b(\w+)( \1\b)+', r'\1', doc)
    doc = doc.replace('\n', ' ')
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = doc.strip(' ')
    #Mengubah menjadi huruf kecil
    doc = doc.lower()
    #Text Normalization
    doc = informal_to_formal_indo(doc)
    #Punctuation Removal+Menghapus Angka
    doc = doc.translate(str.maketrans('', '', string.punctuation + string.digits))
    #Whitespace Removal
    doc = doc.strip()
    #Tokenization
    doc = tokenizer.tokenize(doc)
    doc_token1 = [word for word in doc]
    #Stopwords Removal
    doc_token2 = [word for word in doc_token1 if word not in stopword.get_stopword()]
    #Lemmatization
    doc_token3 = [lemmatizer.lemmatize(word) for word in doc_token2]
    return doc_token3


# %% Apply text pre-processing
review['preprocessing'] = review['content'].apply(my_tokenizer)
print('Preprocessing Result : \n')
print(review[['content', 'preprocessing']])

review1 = review[['content', 'preprocessing']]

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(ulasan) :
  do = []
  for w in ulasan:
    dt = stemmer.stem(w)
    do.append(dt)

  d_clean = []
  d_clean = " ".join(do)
  return d_clean

review['stemming_ulasan'] = review['preprocessing'].apply(stemming)
print('Stemming Result : \n')
# print(review[['stemming_ulasan']])

lexicon_positive = pd.read_excel('Dataset/kamus_positif.xlsx')
lexicon_positive_dict = {}
for index, row in lexicon_positive.iterrows():
    if row[0] not in lexicon_positive_dict:
        lexicon_positive_dict[row[0]] = row[1]

lexicon_negative = pd.read_excel('Dataset/kamus_negatif.xlsx')
lexicon_negative_dict = {}
for index, row in lexicon_negative.iterrows():
    if row[0] not in lexicon_negative_dict:
        lexicon_negative_dict[row[0]] = row[1]

def sentiment_analysis_lexicon_indonesia(ulasan):
    score = 0
    for word in ulasan:
        if (word in lexicon_positive_dict):
            score = score + lexicon_positive_dict[word]
    for word in ulasan:
        if (word in lexicon_negative_dict):
            score = score + lexicon_negative_dict[word]
    sentimen=''
    if (score > 0):
        sentimen = 'positif'
    elif (score < 0):
        sentimen = 'negatif'
    else:
        sentimen = 'netral'
    return score, sentimen

results = review['preprocessing'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
review['label'] = results[0]

# Masih hasil Copy Paste
# data['sentimen'] = results[1] 
# data

review['label'] = results[1]
dataSentimen = review
data_inset = review

data_inset[['content', 'preprocessing', 'label']]

data = review[['stemming_ulasan', 'label']]
data

data['label'].value_counts()

data['label'].value_counts().plot.pie(autopct='%.2f')

data.replace(to_replace='negatif', value=0, inplace=True)
data.replace(to_replace='positif', value=1, inplace=True)
data.replace(to_replace='netral', value=2, inplace=True)
data.head(10)

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data, test_size=0.2)
df_val, df_test = train_test_split(df_test, test_size=0.5)
df_train.shape, df_test.shape, df_val.shape
print('Training data shape:', df_train.shape)
print('Validation data shape:', df_val.shape)
print('Test data shape:', df_test.shape)