# Business-Document-Processing-and-Intelligent-Query-Answering-with-OpenAI-and-Pinecone

Leverage OpenAI and Pinecone to transform document management by processing, embedding, and efficiently answering user queries using advanced AI techniques.

This project demonstrates how to process documents, create embeddings, and answer queries using OpenAI, Pinecone, and LangChain libraries. The system processes a document, splits it into manageable chunks, creates embeddings, stores them in Pinecone, and answers user queries by retrieving relevant information from Pinecone.


![Screenshot 2024-07-24 211839](https://github.com/user-attachments/assets/dc125cd4-efdc-466c-894f-6604bd6cd781)



## Table of Contents
Prerequisites
Installation
Configuration
Usage
Project Structure
Detailed Explanation
Initialize Pinecone
Load and Process Document
Create Embeddings
Answer Queries

## Prerequisites
Before running this project, ensure you have the following:

Python 3.8 or higher
OpenAI API key
Pinecone API key
Required Python libraries (listed in requirements.txt)

## Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/pinecone-doc-processing.git
cd pinecone-doc-processing
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt

## Configuration
Create a .env file in the root directory of the project with the following content:

makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
Replace your_openai_api_key and your_pinecone_api_key with your actual API keys.

## Usage

Run the main script:

bash
Copy code
python main.py
Follow the prompts to enter your query. Type quit or q to exit the program.

## Project Structure

bash
Copy code
.
├── main.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
main.py: The main script to run the project.
utils.py: Contains utility functions for document processing, creating embeddings, and querying.
requirements.txt: Lists the required Python libraries.
.env: Environment variables for API keys.
README.md: This file.

## Detailed Explanation
Initialize Pinecone
python
Copy code
def initialize_pinecone(api_key):
    return Pinecone(api_key=api_key)
This function initializes the Pinecone environment with the provided API key.

## Load and Process Document

python
Copy code
def load_doc(file_path):
    loader = Docx2txtLoader(file_path)
    document = loader.load()
    return document
python
Copy code
def split_docs(document, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(document)
These functions load a document from a specified file path and split it into smaller chunks for processing.

##Create Embeddings

python
Copy code
def create_embeddings(openai_api_key, documents):
    openai.api_key = openai_api_key
    
    model_name = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai.api_key)
    
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name="dheepakindex",
        embedding=embeddings, 
        namespace="adas"
    )
This function creates embeddings for the document chunks using OpenAI and stores them in Pinecone.

## Answer Queries

python
Copy code
def answer_query(query):
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    knowledge = PineconeVectorStore.from_existing_index(
        index_name="dheepakindex",
        namespace="adas",
        embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever()
    )

    response_with_knowledge = qa.invoke(query).get("result")
    print(response_with_knowledge, "\n")
    
    return response_with_knowledge
This function handles user queries by retrieving relevant information from Pinecone and generating responses using OpenAI's language model.

By following this guide, you will be able to set up and run the document processing and query answering system successfully.
