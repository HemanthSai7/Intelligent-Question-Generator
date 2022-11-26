import re
import logging

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import launch_es,print_answers
from haystack.nodes import FARMReader,TransformersReader,BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TextConverter,PDFToTextConverter,PreProcessor
from haystack.utils import convert_files_to_docs, fetch_archive_from_http
from Reader import PdfReader,ExtractedText

launch_es() # Launches an Elasticsearch instance on your local machine

# Install the latest release of Haystack in your own environment
#! pip install farm-haystack

"""Install the latest main of Haystack"""
# !pip install --upgrade pip
# !pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab,ocr]

# # For Colab/linux based machines
# !wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz
# !tar -xvf xpdf-tools-linux-4.04.tar.gz && sudo cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# For Macos machines
# !wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-mac-4.03.tar.gz
# !tar -xvf xpdf-tools-mac-4.03.tar.gz && sudo cp xpdf-tools-mac-4.03/bin64/pdftotext /usr/local/bin

"Run this script from the root of the project"
# # In Colab / No Docker environments: Start Elasticsearch from source
# ! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.2-linux-x86_64.tar.gz -q
# ! tar -xzf elasticsearch-7.9.2-linux-x86_64.tar.gz
# ! chown -R daemon:daemon elasticsearch-7.9.2

# import os
# from subprocess import Popen, PIPE, STDOUT

# es_server = Popen(
#     ["elasticsearch-7.9.2/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1)  # as daemon
# )
# # wait until ES has started
# ! sleep 30

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

class Connection:
    def __init__(self,host="localhost",username="",password="",index="document"):
        """
        host: Elasticsearch host. If no host is provided, the default host "localhost" is used.

        port: Elasticsearch port. If no port is provided, the default port 9200 is used.

        username: Elasticsearch username. If no username is provided, no username is used.

        password: Elasticsearch password. If no password is provided, no password is used.

        index: Elasticsearch index. If no index is provided, the default index "document" is used.
        """
        self.host=host
        self.username=username
        self.password=password
        self.index=index

    def get_connection(self):
        document_store=ElasticsearchDocumentStore(host=self.host,username=self.username,password=self.password,index=self.index) 
        return document_store

class QAHaystack:
    def __init__(self, filename):
        self.filename=filename

    def preprocessing(self,data):
        """
        This function is used to preprocess the data. Its a simple function which removes the special characters and converts the data to lower case.
        """

        converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
        doc_txt = converter.convert(file_path=ExtractedText(self.filename,'data.txt').save(4,6), meta=None)[0]
        
        converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
        doc_pdf = converter.convert(file_path="data/tutorial8/manibook.pdf", meta=None)[0]

        preprocess_text=data.lower() # lowercase
        preprocess_text = re.sub(r'\s+', ' ', preprocess_text) # remove extra spaces
        return preprocess_text        

    def convert_to_document(self,data):

        """
        Write the data to a text file. This is required since the haystack library requires the data to be in a text file so that it can then be converted to a document.
        """
        data=self.preprocessing(data)
        with open(self.filename,'w') as f:
            f.write(data) 

        """
        Read the data from the text file.
        """
        data=self.preprocessing(data)
        with open(self.filename,'r') as f:
            data=f.read()
        data=data.split("\n") 

        """
        DocumentStores expect Documents in dictionary form, like that below. They are loaded using the DocumentStore.write_documents()

        dicts=[
            {
                'content': DOCUMENT_TEXT_HERE,
                'meta':{'name': DOCUMENT_NAME,...}
            },...
        ]

        (Optionally: you can also add more key-value-pairs here, that will be indexed as fields in Elasticsearch and can be accessed later for filtering or shown in the responses of the Pipeline)
        """
        data_json=[{
            'content':paragraph,
            'meta':{
                'name':self.filename
            }
            } for paragraph in data
        ]        

        document_store=Connection().get_connection()
        document_store.write_documents(data_json)
        return document_store
      

class Pipeline:
    def __init__(self,filename,retriever=BM25Retriever,reader=FARMReader):
        self.reader=reader
        self.retriever=retriever
        self.filename=filename

    def get_prediction(self,data,query):
        """
        Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered. They use some simple but fast algorithm.
        
        Here: We use Elasticsearch's default BM25 algorithm . I'll check out the other retrievers as well.
        """
        retriever=self.retriever(document_store=QAHaystack(self.filename).convert_to_document(data))
    
        """
        Readers scan the texts returned by retrievers in detail and extract k best answers. They are based on powerful, but slower deep learning models.Haystack currently supports Readers based on the frameworks FARM and Transformers.
        """
        reader = self.reader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    
        """
        With a Haystack Pipeline we can stick together your building blocks to a search pipeline. Under the hood, Pipelines are Directed Acyclic Graphs (DAGs) that you can easily customize for our own use cases. To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the ExtractiveQAPipeline that combines a retriever and a reader to answer our questions.
        """
        pipe = ExtractiveQAPipeline(reader, retriever)

        """
        This function is used to get the prediction from the pipeline.
        """
        prediction = pipe.run(query=query, params={"Retriever":{"top_k":10}, "Reader":{"top_k":5}})
        return prediction