from utils import (initialize_pinecone, load_doc, split_docs, create_embeddings, 
                   create_index_if_not_exists,answer_query)
import os

def main():
    # Load API keys and initialize Pinecone
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = initialize_pinecone(pinecone_api_key)

    # Document processing
    file_path = 'E:/Doc qa/sample-business-plan-2015.docx'
    document = load_doc(file_path)
    chunks = split_docs(document)

    # Embeddings and Pinecone operations
    create_embeddings(openai_api_key, chunks)
    print("embeddins created succes")
    exit=["quit","q"]
    while True:
        user_input=input("\n enter your query: ")
        if user_input.lower() in exit:
            print("bye")
            break

        response_with_knowledge=answer_query(user_input)
        



if __name__ == "__main__":
    main()

