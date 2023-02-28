# %load text_preprocessor.py
import re
import string
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk import bigrams
from nltk.util import ngrams 
from collections import Counter
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#importing our models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

class TextPreprocessing():

    
    def __init__(self, dataset=None,txt=None):
        self.dataset = dataset
        self.txt = txt
       
        #self.vector = None
    def only_English(self):
       
        self.dataset.language = self.dataset.text.apply(detect)
        self.dataset = self.dataset[self.dataset.language == 'en'] 
        
        #train['language'] = train['text'].apply(detect)
        #train = train[train['language'] == 'en']

  
    
    def _remove_mentions(self,txt):
        #pattern = "@([a-zA-Z0-9_]{1,20})"
        txt = re.sub("@[A-Za-z0-9_]+","", txt) 
        return txt
        
    def remove_mentions(self):
        self.dataset.text = self.dataset.text.apply(lambda x: self._remove_mentions(x))
        
        
    def _remove_hashtags(self,txt):
        pattern = "#([a-zA-Z0-9_]{1,20})"
        txt = re.sub(pattern, '', txt)
        return txt

    
    def remove_hashtags(self):
        self.dataset.text = self.dataset.text.apply(lambda x: self._remove_hashtags(x))
        return self.dataset.text
        

    def text_lower_case(self):
        self.dataset.text= self.dataset.text.apply(lambda x: x.lower())
        return self.dataset.text
   
            
    def remove_numbers(self):
        self.dataset.text = self.dataset.text.apply(lambda x: re.sub(r'\d+','', x))
        return self.dataset.text

    
    def remove_whitespaces(self):
        self.dataset.text = self.dataset.text.apply(lambda x: x.strip())
        return self.dataset.text
  

    def _remove_punct(self, txt):
        txt = ''.join([char for char in txt if char not in string.punctuation])
        txt = re.sub('[0–9]','', txt)
        return txt

    def remove_punct(self):
        self.dataset.text = self.dataset.text.apply(lambda x: self._remove_punct(x))  
        return self.dataset.text

    
  

        
    def _remove_special_characters(self, txt, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        txt = re.sub(pattern, '', txt)
        return txt

    def remove_special_characters(self,remove_digits=False):
        self.dataset.text = self.dataset.text.apply(lambda x: self._remove_special_characters(x))
        return self.dataset.text
    

    def _remove_stopwords(self,txt):
        stop_words = set(stopwords.words('english'))
        stopword_list = nltk.corpus.stopwords.words('english')
        stopword_list.remove('no')
        stopword_list.remove('not')
        tokens = word_tokenize(txt)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
    
    def remove_stopwords(self):
        self.dataset.text = self.dataset.text.apply(lambda x: self._remove_stopwords(x))
        return self.dataset.text


    def _lemmatize_text(self,txt):
        lemmatizer = WordNetLemmatizer()
        tokens_lemma = word_tokenize(txt)
        tokens_lemma = [lemmatizer.lemmatize(w, 'v') for w in tokens_lemma]
        return ' '.join(tokens_lemma)
    
    def lemmatize_text(self):
        self.dataset.text = self.dataset.text.apply(lambda x: self._lemmatize_text(x))
        return self.dataset.text
        
            
    def get_dataset_text(self):
        return self.dataset.text

          
    def fit(self):
        self.only_English()
       
        self.remove_mentions()
        self.text_lower_case()
        self.remove_numbers()
        self.remove_whitespaces()
        self.remove_punct()
        
        self.remove_hashtags()
        self.remove_special_characters()
        self.remove_stopwords()
        self.lemmatize_text()

      
        #self.get_vector()
        
        return self.get_dataset_text()
        #return self.get_vector()
            

