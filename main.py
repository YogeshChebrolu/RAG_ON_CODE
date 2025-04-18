import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pymongo.mongo_client import MongoClient
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq                                                                                                                                                                                        


load_dotenv()

def load_embeddings():
    model_name = "sentence-trasformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    return embeddings


def connect_pinecone():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("textembed")
        return index
    except Exception as e:
        print("Error occured loading pinecone")


def connect_mongodb():
    try:
        client = MongoClient(host=os.getenv('MONGO_CLENT_URI'))
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print("Error connecting MongoDB database")


def create_vector_store(index, embeddings):
    try:
        return PineconeVectorStore(index=index, embeddings=embeddings)
    except Exception as e:
        print("Error creating pinecone vectorstore")


def relevent_text_chunks(query):
    index = connect_pinecone()
    embeddings = load_embeddings()
    vector_store = create_vector_store()

    retriever = vector_store.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k":3, "score_threshold":0.5}
    )

    results = retriever.invoke(query)
    
    return results


def get_parent_id(results):
    parent_ids = {}
    for i, res in enumerate(results):
        key = f"parent{i}"
        parent_ids[key] = res.metadata['chunk_id']
    return parent_ids


def relevent_context(n, text_chunks, parent_ids):
    chunk_ids = ()
    for item in parent_ids:
        chunk_ids.add(item)

    client = connect_mongodb()

    db = client['code']
    code_collection = db['chunks']

    results = relevent_text_chunks()

    context = ""

    retrieved_code_chunks = []

    for i, res in enumerate(results, start=1):
        text = res.page_content
        text = f"Text Chunks {i}" + text
        chunk_id = res.metadata['chunk_id']

        code_chunks = code_collection.find({{"parent_text_id": chunk_id}})
        code_content = ""

        for j, chunk in enumerate(code_chunks, start=1):
            code_content = code_content + f"code {j} of text chunk {i} code content" + chunk['page_content']

        text += code_content

        context += text
    
    return context 




st.title("Pydantic Document Retrieval system")

query = st.text_input("Enter your question", placeholder="What is pydantic ai?")

if query:
    
    # Retrieve relevant text chunks from pinecone

    # Retrieve relevant code chunks from mongodb

    # Display both code and text chunks in the Interface

    # Combine both text and code chunks

    # Create prompt for llm with user query and  retrieved text and code chunks

    prompt = f"""
        You are a helpful python expert. You will asked the user question in query 
        regarding pydantic ai code documentation.
          The relevent code content and text content related the query 
        from documentation are retrieved from the documentaiton and given to you. 
        So you can answer the questioin with better context.
        User's question : {query}
        Context to the question: {context}
        Answer to user's query by explain the retrieved text and code and context.
          If the retrieved content is not helpfull then explain with your own knowledge
    """

    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro",
        temperature = 0.1,

    )

    # invoke the llm

    # Display the result of the llm to interface
    

    