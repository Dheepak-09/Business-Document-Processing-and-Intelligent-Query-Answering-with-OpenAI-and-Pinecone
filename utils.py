import pinecone
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
import openai
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA 
from langchain_openai import ChatOpenAI
import os

def initialize_pinecone(api_key):
    return Pinecone(api_key=api_key)


def load_doc(file_path):
    loader = Docx2txtLoader(file_path)
    document = loader.load()
    return document

def split_docs(document, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(document)

def create_embeddings(openai_api_key, documents):
    openai.api_key = openai_api_key
    
    model_name = "text-embedding-3-small" 
    embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=openai.api_key  
)  
    
    docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name="dheepakindex",
    embedding=embeddings, 
    namespace="adas"
    )

def create_index_if_not_exists(pc, index_name, dimension):
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=dimension, metric='cosine')

def answer_query(query):
    # Initialize a LangChain object for chatting with the LLM without knowledge from Pinecone.
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("openai_api_key"),
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    # Initialize a LangChain object for retrieving information from Pinecone.
    knowledge = PineconeVectorStore.from_existing_index(
        index_name="dheepakindex",
        namespace="adas",
        embedding=OpenAIEmbeddings(openai_api_key=os.environ["openai_api_key"])
    )

    # Initialize a LangChain object for chatting with the LLM with knowledge from Pinecone. 
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Adjust this to the correct chain type or settings you are using.
        retriever=knowledge.as_retriever()
    )

    # Send the query to the LLM twice, first with relevant knowledge from Pinecone
    # and then without any additional knowledge.
    print("Chat with knowledge:")
    response_with_knowledge = qa.invoke(query).get("result")
    print(response_with_knowledge)
    

    return response_with_knowledge


