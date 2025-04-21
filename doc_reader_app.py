from dotenv import load_dotenv
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st 
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings




load_dotenv()

# OpenAI Key
#os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

#Groq
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# LangChain Settings
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "DOc_Reader"
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

GROQ_API_KEY =  "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"

st.set_page_config(page_title="PDF QnA Chatbot")
st.title("Chat with Your PDFs")



upload_file = st.file_uploader("upload a pdf file(< 1MB)",type="pdf")

if upload_file:
    if upload_file.size > 1 * 1024 * 1024:  # 1MB = 1 * 1024 * 1024 bytes
        st.error("❌ File size exceeds 1MB. Please upload a smaller PDF.")
        st.stop()

if upload_file:
   
    with open("temp.pdf","wb") as f:
        f.write(upload_file.read())

    with st.spinner("📚 Reading and indexing the document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)
        chunks = splitter.split_documents(documents=docs)

        #embeddings = OpenAIEmbeddings()
        #embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = FAISS.from_texts([chunk.page_content for chunk in chunks],embedding=embeddings)
        vectorstore.save_local("faiss_index")

    
    question = st.text_input("Ask a Question about your PDF...")

    if question:
        retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        retriver_docs = retriver.invoke(question)

        context = "\n\n".join(doc.page_content for doc in retriver_docs)

        chat = ChatPromptTemplate.from_messages([
            ("system","You are a helpful AI assistant. Use ONLY the context below to answer."),
            ("user","Context:{context} \n\n Question:{question}")
        ])

        model = ChatGroq(model="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY"))
        #model = ChatOpenAI(model="gpt-4o")
        #llm = ChatOpenAI(model=model,temperature=temperature,max_tokens=max_tokens)


        def generate_response(question,context,model=model,temperature=0.5,max_tokens=512):
            llm = model
            chain = chat | llm | StrOutputParser()
            return chain.invoke({"question":question,"context":context})
        
        with st.spinner("✍️ Generating answer..."):
            response = generate_response(question, context)
            st.markdown("### ✅ Answer")
            st.markdown(response)

        with st.expander("🧾 Proof Documents Used"):
             st.write(context)
           




