import uuid

from llama_stack_client import LlamaStackClient, Agent
from dotenv import load_dotenv
import os

from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.types.agents.turn_create_params import ToolConfig
from tavily import TavilyClient

from llama_stack_client.types import UserMessage

load_dotenv()

client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))



@client_tool
def travily_web_search(query: str):
    """
    Tool to perform web search for latest information. This tool returns array of search results consist of title, url, content.

    :param query: search query
    :return: array of search results, each result contains title, url, content, score.
    """
    client = TavilyClient(os.getenv("TAVILY_SEARCH_API_KEY"))
    response = client.search(
        query=query
    )
    print(f"query: {query}, response: {response}")
    return response


agent = Agent(client=client,
              model=os.getenv("INFERENCE_MODEL_ID"),
              instructions=(
                  "You are a web search assistant, must use websearch tool to look up the most current and precise information available. "
              ),
              tools=[travily_web_search],
              tool_config=ToolConfig(tool_choice="required")
              )


def create_websearch_tool_agent_session(prefix: str):
    return agent.create_session(f"{prefix}-{uuid.uuid4()}")


def chat_with_websearch_tool_agent(session_id: str, user_message: str):
    response = agent.create_turn(
        messages=[UserMessage(role="user", content=user_message)],
        session_id=session_id,
        stream=False
    )
    print(f"user message: {user_message}, response: {response}")
    return response.output_message.content


if __name__ == "__main__":
    chat_with_websearch_tool_agent(create_websearch_tool_agent_session("demo_web_search_tool"),
                                     "is Virat Kohli retired from test cricket?")
