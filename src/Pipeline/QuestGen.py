"""Download important files for the pipeline. Uncomment the following lines if you are running this script for the first time"""
# !wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# !tar -xvf  s2v_reddit_2015_md.tar.gz
# if tar file is already downloaded don't download it again
import os
import urllib.request
import tarfile
if not os.path.exists("models/s2v_reddit_2015_md.tar.gz"):
  print ("Downloading Sense2Vec model")
  urllib.request.urlretrieve(r"https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz",filename=r"models/s2v_reddit_2015_md.tar.gz")
else:
  print ("Sense2Vec model already downloaded")  

reddit_s2v= "models/s2v_reddit_2015_md.tar.gz"
extract_s2v="models"
extract_s2v_folder=reddit_s2v.replace(".tar.gz","")
if not os.path.isdir(extract_s2v_folder):
  with tarfile.open(reddit_s2v, 'r:gz') as tar:
    tar.extractall(f"models/")
else:
  print ("Already extracted")

"""Import required libraries"""

import warnings
warnings.filterwarnings('ignore')

from transformers import T5ForConditionalGeneration,T5Tokenizer

import streamlit as st
from sense2vec import Sense2Vec

@st.cache(allow_output_mutation=True)
def cache_models(paths2v,pathT5cond,pathT5):
    s2v = Sense2Vec().from_disk(paths2v)
    question_model = T5ForConditionalGeneration.from_pretrained(pathT5cond)
    question_tokenizer = T5Tokenizer.from_pretrained(pathT5)
    return (s2v,question_model,question_tokenizer)   
s2v,question_model,question_tokenizer=cache_models("models/s2v_old",'ramsrigouthamg/t5_squad_v1','t5-base')


"""Filter out same sense words using sense2vec algorithm"""

def filter_same_sense_words(original,wordlist):
  filtered_words=[]
  base_sense =original.split('|')[1] 
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def sense2vec_get_words(topn,input_keyword):
  word=input_keyword
  output=[]
  required_keywords=[]
  output = []
  try:
    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=topn)
    for i in range(len(most_similar)):
        required_keywords.append(most_similar[i])
    output = filter_same_sense_words(sense,required_keywords)
    print (f"Similar:{output}")
  except:
    output =[]

  return output

"""T5 Question generation"""
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_question(sentence,answer):
  text = f"context: {sentence} answer: {answer} </s>"
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question
