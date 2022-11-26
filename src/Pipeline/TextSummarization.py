import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import numpy as np
from nltk.tokenize import sent_tokenize

class T5_Base:
    def __init__(self,path,device,model_max_length):
        self.model=T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer=T5Tokenizer.from_pretrained(path,model_max_length=model_max_length)
        self.device=torch.device(device)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    def preprocess(self,data):
        preprocess_text=data.strip().replace('\n',' ')
        return preprocess_text

    def post_process(self,data):
        final=""
        for sent in sent_tokenize(data):
            sent=sent.capitalize()
            final+=sent+" "+sent
        return final    

    def getSummary(self,data):
        data=self.preprocess(data)
        t5_prepared_Data="summarize: "+data
        tokenized_text=self.tokenizer.encode_plus(t5_prepared_Data,max_length=512,pad_to_max_length=False,truncation=True,return_tensors='pt').to(self.device)
        input_ids,attention_mask=tokenized_text['input_ids'],tokenized_text['attention_mask']
        summary_ids=self.model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)

        output=[self.tokenizer.decode(ids,skip_special_tokens=True) for ids in summary_ids]
        summary=output[0]
        summary=self.post_process(summary)
        summary=summary.strip()
        return summary


