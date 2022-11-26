from TextSummarization import T5_Base

import spacy
import torch
from transformers import BertTokenizer, BertModel
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
spacy.load() returns a language model object containing all components and data needed to process text. It is usually called nlp. Calling the nlp object on a string of text will return a processed Doc
"""
nlp = spacy.load("en_core_web_sm") #spacy's trained pipeline model

from warnings import filterwarnings as filt
filt('ignore')

class QuestionGenerator:
    def __init__(self,path,device,model_max_length):
        self.model=T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer=AutoTokenizer.from_pretrained(path,model_max_length=model_max_length)
        self.device=torch.device(device)

    def preprocess(self,data):
        preprocess_text=data.strip().replace('\n','')
        return preprocess_text    

    def gen_question(self,data,answer):
        data=self.preprocess(data)
        t5_prepared_data=f'context: {data} answer: {answer}'
        encoding=self.tokenizer.encode_plus(t5_prepared_data,max_length=512,pad_to_max_length=True,truncation=True,return_tensors='pt').to(self.device)
        input_ids,attention_mask=encoding['input_ids'],encoding['attention_mask']
        output=self.model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        num_beams=4,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=512,
                                        early_stopping=True)

        dec=[self.tokenizer.decode(ids,skip_special_tokens=True) for ids in output]
        Question=dec[0].replace("question:","").strip()                              
        return Question
class KeywordGenerator:
    def __init__(self,path,device):
        self.bert_model=BertModel.from_pretrained(path)
        self.bert_tokenizer=BertTokenizer.from_pretrained(path)
        self.sentence_model=SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.device=torch.device(device)

    def get_embedding(self):
        """
        Token Embedding 
        txt = '[CLS] ' + doc + ' [SEP]' where CLS (used for classification task) is the token for the start of the sentence and SEP is the token for the end of the sentence and doc is the document to be encoded. 
        Ex: Sentence A : Paris is a beautiful city.
            Sentence B : I love Paris.
            tokens =[[cls] , Paris, is , a , beautiful , city ,[sep] , I , love , Paris ]
            Before feeding the tokens to the Bert we convert the tokens into embeddings using an embedding layer called token embedding layer.
        """
        tokens=self.bert_tokenizer.tokenize(txt)
        token_idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        """
        Segment Embedding
        Segment embedding is used to distinguish between the two gives sentences.The segment embedding layer returns only either of the two embedding EA(embedding of Sentence A) or EB(embedding of Sentence B) i.e if the input token belongs to sentence A then EA else EB for sentence B.
        """
        segment_ids=[1]*len(token_idx) #This is the segment_ids for the document. [1]*len(token_idxs) is a list of 1s of length len(token_idxs).

        torch_token = torch.tensor([token_idx])
        torch_segment = torch.tensor([segment_ids])
        return self.bert_model(torch_token,torch_segment)[-1].detach().numpy() # 

    def get_posTags(self,context):
        """This function returns the POS tags of the words in the context. Uses Spacy's POS tagger"""
        doc=nlp(context)
        doc_pos=[document.pos_ for document in doc]
        return doc_pos,context.split()

    def get_sentence(self,context):
        """This function returns the sentences in the context. Uses Spacy's sentence tokenizer"""
        doc=nlp(context)
        return list(doc.sents)

    def get_vector(self,doc):
        """
        Machines cannot understand characters and words. So when dealing with text data we need to represent it in numbers to be understood by the machine. Countvectorizer is a method to convert text to numerical data.
        """
        stop_words="english" #This is the list of stop words that we want to remove from the text
        n_gram_range=(1,1) # This is the n-gram range. (1,1)->(unigram,unigram), (1,2)->(unigram,bigram), (1,3)->(unigram,trigram), (2,2)->(bigram,bigram) etc.
        df=CountVectorizer(stop_words=stop_words,ngram_range=n_gram_range).fit([doc])
        return df.get_feature_names() #This returns the list of words in the text.

    def get_key_words(self,context,module_type='t'):
        """
        module_type: 't' for token, 's' for sentence, 'v' for vector
        """
        keywords=[]
        top_n=5
        for txt in self.get_sentence(context):
            keyword=self.get_vector(str(txt))
            print(f'vectors: {keyword}')
            if module_type=='t':
                doc_embedding=self.get_embedding(str(txt))
                keyword_embedding=self.get_embedding(' '.join(keyword))
            else:
                doc_embedding=self.sentence_model.encode([str(txt)])
                keyword_embedding=self.sentence_model.encode(keyword)

            distances=cosine_similarity(doc_embedding,keyword_embedding)
            print(distances)
            keywords+=[(keyword[index],str(txt)) for index in distances.argsort()[0][-top_n:]]

        return keywords     

txt = """Enter text"""
for ans, context in KeywordGenerator('bert-base-uncased','cpu').get_key_words(txt,'st'):
  print(QuestionGenerator('ramsrigouthamg/t5_squad_v1','cpu',512).gen_question(context, ans))
  print()

        



            

