import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


load_dotenv()


class ChatbotResponse(BaseModel):
    response: str
    topic: str
    summary: str
    tool_used: list[str]
    source: list[str]


parser = PydanticOutputParser(pydantic_object=ChatbotResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides information on various topics. Wrap the output in this format and provide no other text.\n{format_instructions}",
        ),
        ("user", "Please provide a response to the following query: {query}"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())


def run_chatbot() -> None:
    llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_tool_calling_agent(llm=llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

    raw_response = agent_executor.invoke({"query": "What is the capital of France?"})
    print(raw_response)


if __name__ == "__main__":
    run_chatbot()
