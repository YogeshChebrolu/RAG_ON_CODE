import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pymongo.mongo_client import MongoClient
from langchain_groq import ChatGroq

load_dotenv()

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

def connect_pinecone():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("textembed")
        return index
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

def connect_mongodb():
    try:
        client = MongoClient(os.getenv('MONGO_CLENT_URI'))
        client.admin.command('ping')
        st.success("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

def create_vector_store(index, embeddings):
    try:
        return PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {e}")
        return None

def relevant_text_chunks(query, vector_store, k=2):
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.5}
        )
        results = retriever.invoke(query)
        return results
    except Exception as e:
        st.error(f"Error retrieving text chunks: {e}")
        return []

def get_parent_ids(results):
    parent_ids = set()
    for res in results:
        parent_ids.add(res.metadata['chunk_id'])
    return parent_ids

def get_relevant_context(results, mongodb_client):
    if not mongodb_client:
        return "Could not connect to MongoDB to retrieve code chunks."
    
    context = ""
    
    try:
        db = mongodb_client['code']
        code_collection = db['chunks']
        
        for i, res in enumerate(results, start=1):
            text = res.page_content
            context += f"Text Chunk {i}: {text}\n\n"
            
            chunk_id = res.metadata['chunk_id']
            code_chunks = code_collection.find({"parent_text_id": chunk_id})
            
            for j, chunk in enumerate(code_chunks, start=1):
                code_content = chunk['page_content']
                context += f"Code {j} for Text Chunk {i}: {code_content}\n\n"
        
        return context
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {e}"

# Main Streamlit app
st.title("Pydantic Document Retrieval System")

query = st.text_input("Enter your question", placeholder="What is pydantic ai?")

if query:
    with st.spinner("Processing your query..."):
        # Set up connections and embeddings
        embeddings = load_embeddings()
        pinecone_index = connect_pinecone()
        mongodb_client = connect_mongodb()
        
        if not pinecone_index or not embeddings:
            st.error("Could not connect to Pinecone or load embeddings.")
        else:
            # Create vector store and retrieve relevant text chunks
            vector_store = create_vector_store(pinecone_index, embeddings)
            if vector_store:
                results = relevant_text_chunks(query, vector_store, k=2)
                
                if results:
                    # Get relevant context combining text and code chunks
                    context = get_relevant_context(results, mongodb_client)
                    
                    # Display retrieved chunks
                    with st.expander("Retrieved Context"):
                        st.write(context)
                    
                    # Create prompt for LLM
                    prompt = f"""
                    You are a helpful Python expert. You will answer the user's question about pydantic AI code documentation.
                    The relevant code content and text content related to the query from documentation are retrieved and given to you below.
                    
                    User's question: {query}
                    
                    Context to the question: {context}
                    
                    Answer the user's query by explaining the retrieved text and code in context.
                    If the retrieved content is not helpful, then explain with your own knowledge about pydantic.
                    """
                    
                    # Invoke the LLM
                    try:
                        llm = ChatGroq(
                            api_key=os.getenv("GROQ_API_KEY"),
                            model="llama3-70b-8192",
                            temperature=0.1,
                        )
                        
                        response = llm.invoke(prompt)
                        
                        # Display the result
                        st.subheader("Answer")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error invoking LLM: {e}")
                        st.write("Please check your API key and model configuration.")
                else:
                    st.warning("No relevant text chunks found for your query.")