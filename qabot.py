from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as genparams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

## Warning suppression
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## Global variables
url = "your url"
project_id = "project ID"
llm_model_id = 'meta-llama/llama-3-3-70b-instruct' # other model 'mistralai/mixtral-8x7b-instruct-v01'
embed_model_id = 'ibm/slate-125m-english-rtrvr'

## Model Definiton
def model():
    parameters = {
        genparams.TEMPERATURE : 0.5,
        genparams.MAX_NEW_TOKENS: 256
    }
    watsonx_llm = WatsonxLLM(
        model_id = llm_model_id,
        url = url,
        project_id = project_id,
        params = parameters,
    )
    return watsonx_llm

## Doc loader
def doc_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_doc = loader.load()
    return loaded_doc

## Text Splitter
def text_split(data):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
    )
    chunks = text_split.split_documents(data)
    return chunks

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id= embed_model_id,
        url= url,
        project_id= project_id,
        params=embed_params,
    )
    return watsonx_embedding

## Vector DB
def vec_db(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = doc_loader(file)
    chunks = text_split(splits)
    vectordb = vec_db(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA
def retriever_qa(file, query):
    llm = model()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type = "stuff",
                                      retriever=retriever_obj,
                                      return_source_documents=False)
    response = qa.invoke({"query": query})
    return response['result']

# Interface Definition
rag_app = gr.Interface(
    fn = retriever_qa,
    allow_flagging = 'auto',
    inputs = [
        gr.File(label= "Upload PDF file", file_count = "single", file_types = [".pdf"], type = "filepath"),
        gr.Textbox(label = "Input Query", lines = 2, placeholder = "Type your query here.")
    ],
    outputs = gr.Textbox(label = "Query Result"),
    title = "RAG based QA Bot"
)

# Launcher
rag_app.launch(server_name = "127.0.0.1", server_port = 7860)
