import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms.cohere import Cohere
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback
import pickle
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
import os

COHERE_API_KEY = os.getenv('COHERE_API_KEY') or 'ceBXM6Q8ggpUzZGzcNLZfbE7tEq9mF4qxC58UUrz'
HUGGING_FACE_API = os.getenv("HUGGING_FACE") or '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'

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is not set")
if not HUGGING_FACE_API:
    raise ValueError("HUGGING_FACE_API is not set")

embedding_model = "sentence-transformers/all-mpnet-base-v2"
llm_model = "google/flan-t5-xl"
llm = Cohere(cohere_api_key=COHERE_API_KEY)

def summarize():
    user_input_pdf = st.file_uploader("Upload your pdf:", type='pdf')
    if user_input_pdf is not None:
        pdf_file = PdfReader(user_input_pdf)
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
        
        template = '''Hi, I want you to act as AI Assistant for high school students who want to understand the book. 
        The text of book is {text}. Please keep your language in layman terms. Can you please summarize this book in 
        20 words? '''
        prompt = PromptTemplate(input_variables=["text"], template=template)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        st.markdown("Summary of PDF is: ")
        if len(text) > 4000:
            contract_text = text[:4000]
            st.write(llm_chain.run(contract_text))
        else:
            st.write(llm_chain.run(text))
        add_vertical_space(2)
        
        root_embedding_path = 'src\\embeddings'
        store_name = user_input_pdf.name[:-4]
        embedding_root_path = os.path.join(root_embedding_path, store_name)
        if os.path.exists(f'{embedding_root_path}.pkl'):
            with open(f"{embedding_root_path}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            embeddings = HuggingFaceHubEmbeddings(repo_id=embedding_model, huggingfacehub_api_token=HUGGING_FACE_API)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{embedding_root_path}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

summarize()
