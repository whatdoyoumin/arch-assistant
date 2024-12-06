import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import pickle 

import streamlit as st

from util.utility import check_password

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

location_list = [
    {'name':'Singapore Cricket Club', 'short_name':'scc'},
    {'name':'Fort Canning', 'short_name':'fortcanning'},
    {'name':'Fort Canning Spice Garden', 'short_name':'spicegarden'},
    {'name':'Pulau Saigon', 'short_name':'pulausaigon'},
    {'name':'St. Andrews', 'short_name':'sta'},
    {'name':'Istana Kampung Gelam', 'short_name':'ikg'},
    {'name':'Empress Place 1998', 'short_name':'emp'},
    {'name':'Empress Place 2015', 'short_name':'emp2015'},
    {'name':'The Temasek Wreck', 'short_name':'temasekwreck'},
]


# Retrieval Augmentation Generation
#load vectorstore
@st.cache_resource
def load_vectorstore():
    #call openai embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local("./faiss_index", embeddings_model, allow_dangerous_deserialization=True)

    with open("./faiss_index/metadata.pkl", "rb") as f:
        vector_store.docstore.__dict__ = pickle.load(f)
        
    return vector_store 
vector_store = load_vectorstore()

#filter vectorstore based on location of interest
def filter_vectorstore(vector_store, loc_int):
    filter_dict = {"source":"./sitereports/"+loc_int+".txt"}
    
    vector_store_filter = vector_store.as_retriever(
        search_kwargs={"filter": filter_dict}
    )
    return vector_store_filter

#question & answer
def qna(vector_store_filter, user_qn):
    # Build prompt
    template = """Only use provided context to answer the question at the end.
    If there is no context provided or you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum. Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model='gpt-4o-mini'),
        retriever=vector_store_filter,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    output = qa_chain.invoke(user_qn)
    return output


# Streamlit app
def query_app(location_list, vector_store):
    # Do not continue if check_password is not True.  
    if not check_password():  
        st.stop()

    st.title("Query Archaeological Site Reports")
    st.write("For information on the full site reports, please refer to NUS E-Press at https://epress.nus.edu.sg/sitereports/.")

    #list of long and short names of sites
    names = []
    short_names = []
    for loc in location_list:
        names.append(loc["name"])
        short_names.append(loc["short_name"])

    #selection box to select site
    location_s = st.selectbox("Select Archaeological Site", names)
    #corresponding short name of selected site
    loc_s = short_names[names.index(location_s)]

    #user query
    user_query = st.text_input("Ask me qns about the site", "E.g. What are the artefacts found in this site?")
    if (st.button('Submit')):
        query = user_query.title()
        #call RAG pipeline and display response
        vector_store_filter = filter_vectorstore(vector_store, loc_s)
        answer = qna(vector_store_filter, query)

        st.text("Answer:")
        st.write(answer["result"])

        st.text("Sources of Information:")
        st.write(answer["source_documents"])

query_app(location_list, vector_store)