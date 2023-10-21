from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your pdf")
    st.header("Chat with your PDF 🗨️ ")

    #Upload the pdf file
    pdf= st.file_uploader("Upload pdf here ",type="pdf")

    #extract the text from the pdf file
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        #split text recived into equal chunks
        text_spliiter = CharacterTextSplitter(  
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spliiter.split_text(text)

        #Convert text chunks into embeddings

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question= st.text_input("Ask your PDF a question")
        if user_question:
            docs=knowledge_base.similarity_search(user_question)
            chain= load_qa_chain(
                llm=OpenAI(),
                chain_type="stuff"
            )
            response=chain.run(input_documents=docs,question=user_question)
            
            st.write(response)



if __name__ =="__main__":
    main()