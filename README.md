---
title: Question Generator
emoji: 🔑
colorFrom: yellow
colorTo: yellow
sdk: streamlit
sdk_version: "1.10.0"
app_file: app.py
pinned: false
---

# Internship-IVIS-labs

-  The *Intelligent Question Generator* app is an easy-to-use interface built in Streamlit which uses [KeyBERT](https://github.com/MaartenGr/KeyBERT), [Sense2vec](https://github.com/explosion/sense2vec), [T5](https://huggingface.co/ramsrigouthamg/t5_paraphraser)
-  It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers](https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
- [sense2vec](https://github.com/explosion/sense2vec) (Trask et. al, 2015) is a nice twist on word2vec that lets you learn more interesting and detailed word vectors.

## Repository Breakdown
### src Directory
---
- `src/Pipeline/QAhaystack.py`: This file contains the code of question answering using [haystack](https://haystack.deepset.ai/overview/intro).
- `src/Pipeline/QuestGen.py`: This file contains the code of question generation.
- `src/Pipeline/Reader.py`: This file contains the code of reading the document.
- `src/Pipeline/TextSummariztion.py`: This file contains the code of text summarization.
- `src/PreviousVersionCode/context.py`: This file contains the finding the context of the paragraph. 
- `src/PreviousVersionCode/QuestionGenerator.py`: This file contains the code of first attempt of question generation.

## Installation
```shell
$ git clone https://github.com/HemanthSai7/Internship-IVIS-labs.git
```
```shell
$ cd Internship-IVIS-labs
```
```python
pip install -r requirements.txt
```
- For the running the app for the first time locally, you need to uncomment the the lines in `src/Pipeline/QuestGen.py` to download the models to the models directory.

```python
streamlit run app.py
```
- Once the app is running, you can access it at http://localhost:8501
```shell
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.103:8501
```

## Tech Stack Used
![image](https://img.shields.io/badge/Sense2vec-EF546D?style=for-the-badge&logo=Explosion.ai&logoColor=white)
![image](https://img.shields.io/badge/Spacy-09A3D5?style=for-the-badge&logo=spaCy&logoColor=white)
![image](https://img.shields.io/badge/Haystack-03AF9D?style=for-the-badge&logo=Haystackh&logoColor=white)
![image](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![image](https://img.shields.io/badge/PyTorch-D04139?style=for-the-badge&logo=pytorch&logoColor=white)
![image](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![image](https://img.shields.io/badge/Pandas-130654?style=for-the-badge&logo=pandas&logoColor=white)
![image](https://img.shields.io/badge/matplotlib-b2feb0?style=for-the-badge&logo=matplotlib&logoColor=white)
![image](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![image](https://img.shields.io/badge/Streamlit-EA6566?style=for-the-badge&logo=streamlit&logoColor=white)

## Timeline
### Week 1-2:
#### Tasks
- [x] Understanding and brushing up the concepts of NLP.
- [x] Extracting images and text from a pdf file and storing it in a texty file.
- [x] Exploring various open source tools for generating questions from a given text.
- [x] Read papers related to the project (Bert,T5,RoBERTa etc).
- [x] Summarizing the extracted text using T5 base pre-trained model from the pdf file.

### Week 3-4:
#### Tasks
- [x] Understanding the concept of QA systems.
- [x] Created a basic script for generating questions from the text.
- [x] Created a basic script for finding the context of the paragraph.

### Week 5-6:
#### Tasks

- [x] Understanding how Transformers models work for NLP tasks Question answering and generation
- [x] Understanding how to use the Haystack library for QA systems.
- [x] Understanding how to use the Haystack library for Question generation.
- [x] PreProcessed the document for Haystack QA for better results .

### Week 7-8:
#### Tasks
- [x] Understanding how to generate questions intelligently.
- [x] Explored wordnet to find synonyms
- [x] Used BertWSD for disambiguating the sentence provided.
- [x] Used KeyBERT for finding the keywords in the document.
- [x] Used sense2vec for finding better words with high relatedness for the keywords generated.

### Week 9-10:
#### Tasks
- [x] Create a streamlit app to demonstrate the project.
