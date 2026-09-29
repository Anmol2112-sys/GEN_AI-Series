from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st


llm=ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search=GoogleSerperAPIWrapper()
tools=[search.run]
memory=MemorySaver()

if "memory" not in st.session_state:
    st.session_state.memory=MemorySaver()
    st.session_state.history=[{"role":"user","content":"query"},{"role":"ai","content":"query"}]

agent=create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt="You are a amazing ai agent and can search on google as well"
)

print(st.session_state.memory)
st.subheader("Flash-Answers at the speed of thought")

for message in st.session_state.history:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)
    
    
query=st.chat_input("Ask Anything")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    
    response=agent.stream(
        {"messages":[{"role":"user","content":"Who is Pm of India?"}]},
        {"configurable":{"thread_id":"1"}},
        stream_mode="messages"
)

    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()
        
        
        message=""
        
        for chunk in response:
            message=message+chunk[0].content
            space.write(message)
            
        st.session_state.history.append({"role":"ai","content":message})
    