# RAG-Based QA Bot using LangChain and IBM watsonx.ai

This repository contains a **Retrieval-Augmented Generation (RAG) based Question Answering (QA) Bot** that leverages LangChain, IBM watsonx.ai Large Language Models (LLMs), and vector databases to answer questions from PDF documents. The bot integrates document loading, text splitting, embedding generation, vector storage, retrieval, and a web interface built with Gradio for an end-to-end document QA solution.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [References](#references)

## Project Overview

Manually searching large volumes of PDFs for answers can be tedious and inefficient. This project automates the process using LangChain combined with IBM watsonx.ai LLMs:

- **Load and parse PDF documents**
- **Split content into manageable chunks**
- **Convert chunks into embeddings using IBM Slate embeddings**
- **Store embeddings in a Chroma vector database**
- **Retrieve relevant chunks based on user query**
- **Answer questions using an LLM with retrieval-augmented generation**
- **Provide a clean and interactive UI via Gradio**


## Features

- **Supports multiple LLM foundational models** — e.g., Meta LLaMA 3 70B, Mixtral 8x7B
- Efficient **document loading and chunking** via PyPDFLoader and RecursiveCharacterTextSplitter
- Embedding generation using **IBM Slate 125M English model**
- Vector storage and fast retrieval powered by **ChromaDB**
- Automatic **question-answering chains** from LangChain for seamless RAG
- **User-friendly Gradio interface** for uploading PDFs and querying

## Note: You are required to have your own API-key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akavkl/RAG-Based-Question-Answering-Bot-using-IBM-Watsonx-and-LangChain.git
cd RAG-Based-Question-Answering-Bot-using-IBM-Watsonx-and-LangChain
```

2. Create and activate a Python virtual environment:
```bash
python3.11 -m venv my_env
source my_env/bin/activate
```

3. Install dependencies:
```bash
pip install gradio==4.44.0 ibm-watsonx-ai==1.1.2 langchain==0.2.11 langchain-community==0.2.10 langchain-ibm==0.1.11 chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1
```


## Usage

1. Launch the Gradio application:
```bash
python3.11 qabot.py
```

2. Open your browser and navigate to `http://127.0.0.1:7860`
3. Upload a PDF file and type your question.
4. The bot will retrieve relevant information and answer your query using the content of the document.

## Code Structure

- `qabot.py` — main Python script containing the following components:
    - Importing necessary libraries
    - Suppressing warnings for clean logs
    - Defining the Watsonx LLM model with configurable parameters
    - Loading PDF documents using PyPDFLoader
    - Splitting texts into manageable chunks for embeddings
    - Generating embeddings with IBM Slate 125M English model
    - Creating a Chroma vector database to store embeddings
    - Defining a retriever for similarity search
    - Building a question-answering chain leveraging LangChain `RetrievalQA`
    - Gradio UI interface definition and launching


## How It Works

### 1. Loading Documents

Using `PyPDFLoader`, the PDF is broken down into raw text data that can be processed.

### 2. Splitting Text

`RecursiveCharacterTextSplitter` slices the text into chunks of 1000 characters with an overlap of 100 to preserve context.

### 3. Embedding Chunks

Chunks are converted to dense vector embeddings via IBM watsonx's Slate 125M embeddings model.

### 4. Vector Database

Embeddings are stored and indexed in a `Chroma` vector store for similarity search.

### 5. Retriever

For a new query, the retriever searches the vector store for the most relevant chunks.

### 6. Question Answering

The `RetrievalQA` chain executes using Watsonx LLM (`meta-llama/llama-3-3-70b-instruct` by default) to generate an answer using retrieved documents.

### 7. User Interface

Gradio provides a simple UI where users can upload PDFs, type questions, and receive answers instantly.

## References

- LangChain documentation: https://langchain.com
- IBM watsonx.ai documentation: https://cloud.ibm.com/docs/watsonx
- Chroma Vector Database: https://www.trychroma.com
- Gradio: https://gradio.app

Feel free to raise issues or pull requests to improve the QA bot template!

<div style="text-align: center">⁂</div>

[^1]: download.pdf

