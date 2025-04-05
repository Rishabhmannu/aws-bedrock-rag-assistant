import json
import os
import sys
import boto3
import streamlit as st
import time
from botocore.config import Config

# Updated imports to use the newest packages
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Client
# Bedrock Client (fixed)
bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name="us-east-1",
    config=Config(
        retries={
            'max_attempts': 10,
            'mode': 'adaptive'
        }
    )
)
# Create embedding client with direct API call method to avoid throttling
class CustomBedrockEmbeddings(BedrockEmbeddings):
    def _embedding_func(self, text):
        """Embed a text using the Bedrock API."""
        # Custom implementation using the Titan embedding model format
        payload = json.dumps({"inputText": text})
        
        try:
            response = self.client.invoke_model(
                body=payload,
                modelId=self.model_id,
                accept="*/*",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")
        except Exception as e:
            print(f"Error in embedding: {str(e)}")
            # Add backoff for throttling
            if "ThrottlingException" in str(e):
                sleep_time = 5  # seconds
                print(f"Throttling detected, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                # Retry once after sleeping
                response = self.client.invoke_model(
                    body=payload,
                    modelId=self.model_id,
                    accept="*/*",
                    contentType="application/json"
                )
                response_body = json.loads(response.get("body").read())
                return response_body.get("embedding")
            else:
                raise e

# Create embedding client
bedrock_embeddings = CustomBedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock,
)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    # Reduced chunk size to avoid throttling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Significantly reduced from 10000
        chunk_overlap=200  # Reduced from 1000
    )
    
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    try:
        st.info(f"Processing {len(docs)} document chunks...")
        
        # Process in small batches with delays between each
        batch_size = 3  # Very small batch size to avoid throttling
        
        # Create empty FAISS index first
        vectorstore_faiss = None
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            batch_end = min(i + batch_size, len(docs))
            batch_docs = docs[i:batch_end]
            
            st.info(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}")
            
            # Create or update vectorstore
            if vectorstore_faiss is None:
                vectorstore_faiss = FAISS.from_documents(
                    batch_docs,
                    bedrock_embeddings
                )
            else:
                # Add documents to existing vectorstore
                vectorstore_faiss.add_documents(batch_docs)
            
            # Save after each batch to preserve progress
            if vectorstore_faiss is not None:
                vectorstore_faiss.save_local("faiss_index")
            
            # Add delay between batches unless it's the last batch
            if batch_end < len(docs):
                time.sleep(3)  # 3 second delay between batches
        
        return vectorstore_faiss
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise e

def get_claude_llm():
    # Create Claude Model with proper chat interface
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={
            'max_tokens': 512,
            'temperature': 0.7
        }
    )
    return llm
  
  
prompt_template = """
Human: Use the context below to answer the question. Provide detailed explanations in 250+ words.
If unsure, say you don't know.

Context: {context}

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    # if st.button("Llama2 Output"):
    #     with st.spinner("Processing..."):
    #         faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
    #         llm=get_llama2_llm()
            
    #         #faiss_index = get_vector_store(docs)
    #         st.write(get_response_llm(llm,faiss_index,user_question))
    #         st.success("Done")

if __name__ == "__main__":
    main()


