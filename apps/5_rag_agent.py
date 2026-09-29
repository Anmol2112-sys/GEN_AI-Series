from dotenv import load_dotenv
load_dotenv()



from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False
    
if "agent" not in st.session_state:
    st.session_state.agent=None
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
    
if "messages" not in st.session_state:
    st.session_state.messages=[]

def process_document(path):
    docs=loader.load()
    ### Load the documents
    loader=PyPDFDirectoryLoader("../data/System-design-Interview-Questions.pdf")
    docs=loader.load()

    ### Split into chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(docs)

    ### embeddings and Vector DB
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    ### Create a agent- tool,llm,prompt
    llm=ChatGroq(model="openai/gpt-oss-20b")
    @tool
    def retrieve_context(query:str):
        """
        Retrieve relevant documents for a given query from the vector database.
        """
        context=""
        
        docs=vector_db.similarity_search(query=query,k=3)
        for doc in docs:
            context=doc.page_content+"\n\n"
            
        return context

    system_prompt="""
    You are an expert in system design. Answer the following question based on the provided context.
    If you don't know the answer, just say that you don't know.
    """

    memory=InMemorySaver()


    agent=create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    st.session_state.document_uploaded=True

    ### upload ui
    if not st.session_state.document_uploaded:
        uploaded=st.file_uploader(label="Select PDF Files", type=["pdf"],accept_multiple_files=True)
        if uploaded:
            with st.spinner("Processing..."):
                path="./doc_files/"
                for file in uploaded:
                    with open(path+file.name,"wb") as f:
                        f.write(file.getvalue())


                process_document(path)
                st.rerun()
    ### chat ui
    
    if st.session_state.document_uploaded and st.session_state.agent:
        for message in st.session_state.messages:
            role=message.get("role")
            content=message.get("content")
            st.chat_message(role).markdown(content)
            
        query=st.chat_input("Ask anything related to uploaded documents...")
        if query:
            st.session_state.message.append({"role":"user","content":query})
            st.chat_message("user").markdown(query)
            response=st.session_state.agent(
                {"messages":[{"role":"user","content":query}]},
                {"configurable":{"thread_id":"1"}}
                
            )
            answer=response["messages"][-1].content
            st.chat_message("ai").markdown(answer)
            st.session_state.messages.append({"role":"ai", "content":answer})
