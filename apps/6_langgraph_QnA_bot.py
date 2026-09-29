from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

class ChatState(BaseModel):
    messages: Annotated[list, add_messages]


llm = ChatGroq(model="openai/gpt-oss-20b")


def ChatBotNode(state: ChatState):

    res = llm.invoke(state.messages)

    return {"messages": [res]}


memory = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chatbot", ChatBotNode)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph = graph.compile(checkpointer=memory)


from IPython.display import Image

Image(graph.get_graph().draw_mermaid_png())


config = {"configurable": {"thread_id": "my-bot-1"}}

while True:

    query = input("User: ")

    if query.lower() in ["quit", "exit", "bye"]:
        print("Thanks for Using!")
        break

    res = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config=config
    )

    ans = res["messages"][-1].content

    print("AI:", ans)