from langchain import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.callbacks import get_openai_callback
import os

#sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - LangChain
    - OpenAI LLM model
 
    ''')
    add_vertical_space(5)
    
    st.write("Upload your PDF and start asking your questions!")
 

load_dotenv()
def main():
   
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        # Add a slider for selecting the number of relevant chunks (k value)
        number_of_chunks = st.slider(
            "Number of relevant chunks", min_value=1, max_value=7, value=2, step=1
        )

        # Add radio buttons for selecting the chain type
        chain_type = st.radio(
            "Chain type",
            options=['stuff', 'map_reduce'],
            index=1  # Index of 'map_reduce' in the options list
        )

        pdf_reader = PdfReader(pdf)
        text = ""
        #break down into pages
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #ensure that each chunk of text fed int othe model is within token limit
        #gpt => 4096  
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)

        #save embeddings onto disk 
        store_name = pdf.name[:-4]
        #check if file already exists
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()
            VectoreStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectoreStore, f)
        
        #Accept queries
        query = st.text_input("Ask questions about your PDF file")
        run_button =st.button('Run')
        if query and run_button:
            #context 
            docs = VectorStore.similarity_search(query=query, k=number_of_chunks) 
            #st.write(docs) 
            llm = ChatOpenAI(model_name='gpt-4')
            if chain_type == 'stuff':
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
            else:        
                #takes a chunk and summarize it 
                map_custom_prompt = '''
                You are an AI/ML engineer working for a technology firm in the environmental space. The company has identified a gap in the market for a tool that aids stakeholders in understanding and navigating the dense and intricate carbon market documents, especially on Verra's carbon credit methodologies and guidelines.
                Summarize the following text in a clear and concise way:
                TEXT:`{text}`
                    Brief Summary:
                '''
                map_prompt_template = PromptTemplate(
                    input_variables=['text'],
                    template = map_custom_prompt
                )
                combine_custom_prompt='''
                You are an AI/ML engineer working for a technology firm in the environmental space. The company has identified a gap in the market for a tool that aids stakeholders in understanding and navigating the dense and intricate carbon market documents, especially on Verra's carbon credit methodologies and guidelines.
                Generate a summary of the following text that includes the following elements:

                * A title that accurately reflects the content of the text.
                * An introduction paragraph that provides an overview of the topic.
                * Bullet points that list the key points of the text.
                * A conclusion paragraph that summarizes the main points of the text.

                Text:`{text}`
                '''
                combine_prompt_template = PromptTemplate(
                    input_variables=['text'],
                    template = combine_custom_prompt
                )
            
                chain = load_summarize_chain (
                        llm=llm,
                        chain_type='map_reduce',
                        map_prompt = map_prompt_template,
                        combine_prompt=combine_prompt_template,
                        verbose=False
                    )
                with get_openai_callback() as cb:
                    split_docs = text_splitter.split_documents(docs)
                    response = chain.run(input_documents=split_docs)
                    print(cb)
            st.write(response)

if __name__ == '__main__':
    main() 