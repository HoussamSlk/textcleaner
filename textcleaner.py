# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:52:36 2020

@author: qihus

v1
"""
from ftfy import fix_text
import fasttext
import warnings
import re
import spacy
import en_core_web_sm
import de_core_news_sm
warnings.filterwarnings("ignore")
PRETRAINED_MODEL_PATH = 'lid.176.ftz' #change path accordingly
model = fasttext.load_model(PRETRAINED_MODEL_PATH)
import pycountry
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
supported_language =['arabic','azerbaijani','danish','dutch','english','finnish','french',\
'german','greek','hungarian','indonesian','italian','kazakh','nepali',\
'norwegian','portuguese','romanian','russian','spanish','swedish','turkish']
    
my_file = open("removable_words.txt", "r")
removable_words = my_file.read().split('\n') #change path if needed
my_file = open("bigram_phrases.txt", "r")
bigram_keywords = my_file.read().split('\n')


class TextCleaner:
    def __init__(self):
        pass
    def __fix_encoding(self,text):
        self.text = fix_text(text)
        
        
    def __detect_language(self,text):
        pred = model.predict(text)
        self.lang = pred[0][0].split("__label__")[1]
        
    def __remove_words(self,text):
        text = [w for w in text.split() if w not in removable_words]
        text = ' '.join(text)
        self.text = text
        
    def __label_bigram_phrases(self,text):  
        phrases = [word for word in bigram_phrases if word in text]
        for phrase in phrases:
            text = text+' '+ phrase.replace(" ", "_")
        self.text = text
        
    def __remove_non_alpha_non_latin(self,text):
        text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '',text)
        text = [w for w in text.split() if w.isalpha()] #keep if contains at least one alpha 
        text = ' '.join(text)
        self.text = text
   
    def __remove_ner(self,text,lang='en'):
        if lang == 'en':
            self.nlp = spacy.load(lang + "_core_web_sm")
        if lang in ['de', 'fr']:
            self.nlp = spacy.load(lang + "_core_news_sm")
        document = self.nlp(text)
        ents = [e.text for e in document.ents]
        for ent in ents:
            if ent in text:
                text = text.replace(ent,'')
        self.text = text
    
    def __only_nv(self, text, lang='en'):
        if lang == 'en':
            self.nlp = spacy.load(lang + "_core_web_sm")
        if lang in ['de', 'fr']:
            self.nlp = spacy.load(lang + "_core_news_sm")
        document = self.nlp(text)
        words = [token.text for token in document if token.pos_ in ['VERB', 'NOUN']]
        self.text =  ' '.join(words)
   
    def __remove_small_words(self,text):
       self.text = re.sub(r'\b\w{1,3}\b','',text) #less than 4 char remove
    
    def __remove_stop_words(self,text):
        text = text.lower()
        if pycountry.languages.get(alpha_2=self.lang) is not None:
            language = pycountry.languages.get(alpha_2=self.lang).name.lower()
            if language in supported_language:
                text = [w for w in text.split() if w not in stopwords.words(language)]
                text = ' '.join(text)
            self.text = text


    def clean(self,text):
        self.__fix_encoding(text)
        self.__detect_language(text)
        self.__remove_non_alpha_non_latin(text)
        self.__remove_stop_words(text)
        #self.__remove_ner(text,self.lang)
        self.__only_nv(text, self.lang)