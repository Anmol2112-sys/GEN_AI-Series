import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

### GET LLM
def get_llm(model_name:str="openai/gpt-oss-20b",temperature=float=0.5):
    api_key=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key
        return llm
    )
    
    
    
#### Research Agent
RESEARCHER_PROMPT=ChatPromptTemplate.from_messages(
 {"role":"system","content":"""
     You are a research assistant AI. Your task is to help users find information on a given topic.
   Use the following steps to complete your task:
   1. Understand the user's query
   2. Search for relevant information
   3. Evaluate the credibility of sources
   4. Present findings clearly and concisely
   5. If unsure, honestly state limitations

   Always prioritize recent, reliable sources and present information objectively.
  """},{
      "role":"user",
      "content":"Topic":{topic},Audience:{audience},{revision-hints}.Write the research outline now."
  }
)

def researcher_agent(llm:ChatGroq,topic:str,audience:str,research:str="",feedback:str="")->str:
    revision_hints=f"The human provided this feedback on your previous draft and asked for these changes:{feedback}.Please apply these changes during writting the blog"
    if not feedback:
        revision_hints="This is your first attempt."
        
    chain=RESEARCHER_PROMPT | llm
    
    result=chain.invoke({
        "topic":topic,
        "audience":audience,
        "research":research,
        "revision-hints":revision_hints
    })

    return result.content

### Writer Agent
WRITER_PROMPT=ChatPromptTemplate.from_messages(
 {"role":"system","content":"""
     You are a research assistant AI. Your task is to help users find information on a given topic.
   Use the following steps to complete your task:
   1. Understand the user's query
   2. Search for relevant information
   3. Evaluate the credibility of sources
   4. Present findings clearly and concisely
   5. If unsure, honestly state limitations
   Always prioritize recent, reliable sources and present information objectively.
  """},{
      "role":"user",
      "content":"""Topic:{topic},Audience:{audience},Research Notes:{research}{revision-hints}.Write the research outline now."
  """}
)

def writer_agent(llm:ChatGroq,topic:str,audience:str,research:str="",feedback:str="")->str:
    revision_hints=f"The human provided this feedback on your previous draft and asked for these changes:{feedback}.Please apply these changes during writting the blog"
    if not feedback:
        revision_hints="This is your first attempt."
        
    chain=WRITER_PROMPT | llm
    
    result=chain.invoke({
        "topic":topic,
        "audience":audience,
        "research":research,
        "revision-hints":revision_hints
    })

    return result.content


### EDITOR AGENT
EDITOR_PROMPT=ChatPromptTemplate.from_messages(
 {"role":"system","content":"""
     You are a research assistant AI. Your task is to help users find information on a given topic.
   Use the following steps to complete your task:
   1. Understand the user's query
   2. Search for relevant information
   3. Evaluate the credibility of sources
   4. Present findings clearly and concisely
   5. If unsure, honestly state limitations
   Always prioritize recent, reliable sources and present information objectively.
  """},{
      "role":"user",
      "content":"""Topic:{topic},Draft:{draft}
      Return the published blog post."
  """}
)

def editor_agent(llm:ChatGroq, topic:str, draft:str)->str:

    chain=EDITOR_PROMPT | llm

    result=chain.invoke({
        "topic":topic,
        "draft":draft,
    })

    return result.content


