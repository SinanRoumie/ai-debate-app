from pydantic_ai import Agent, RunContext, Tool
from typing_extensions import TypedDict
from typing import Annotated
import time
from duckduckgo_search import DDGS

class ResearchInput(TypedDict):
    query: str
    
class DebateMessage(TypedDict):
    agent_type: str
    message: str

message_storage: list[DebateMessage] = []


def safe_duckduckgo_search(query: Annotated[str, "The search query"]) -> str:
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query)
                return "\n".join([r["body"] for r in results])
        except Exception as e:
            print(f"Search failed (attempt {attempt+1}): {e}")
            time.sleep(5)
    return "Search failed after retries."


duckduckgo_tool = Tool(
    name="duckduckgo_search",
    description="Searches the web using DuckDuckGo",
    function=safe_duckduckgo_search
)

## Affirmative Debater ##

debater_a = Agent(
    model="openai:gpt-3.5-turbo",
    system_prompt='You are Debater A. You will affirm the topic.'
)

debater_a_research_agent = Agent(
    name="Research Agent A",
    model="openai:gpt-3.5-turbo",
    system_prompt="You are a research assistant for a debate. You support the pro side. Search the internet to find 10 relevant facts or arguments to support your position.",
    tools=[duckduckgo_tool]
)

@debater_a.system_prompt
async def add_debater_a_data(ctx: RunContext[str]) -> str:
    debate_topic = ctx.deps
    return f"This is the topic you will be debating: {debate_topic!r}"

@debater_a_research_agent.system_prompt
async def add_debater_a_data(ctx: RunContext[str]) -> str:
    debate_topic = ctx.deps
    return f"There is the search topic: {debate_topic!r}"

## Negative Debater ##

debater_n = Agent(
    model="openai:gpt-3.5-turbo",
    system_prompt='You are Debater N. You will negate the topic.'
)

debater_n_research_agent = Agent(
    name="Research Agent N",
    model="openai:gpt-3.5-turbo",
    system_prompt="You are a research assistant for a debate. You support the con side. Search the internet to find 10 relevant facts or arguments to support your position.",
    tools=[duckduckgo_tool]
)

@debater_n.system_prompt
async def add_debater_n_data(ctx: RunContext[str]) -> str:
    debate_topic = ctx.deps
    return f"This is the topic you will be debating: {debate_topic!r}"



@debater_n_research_agent.system_prompt
async def add_debater_n_data(ctx: RunContext[str]) -> str:
    debate_topic = ctx.deps
    return f"There is the search topic: {debate_topic!r}"

## Judge ##
judge = Agent(
    model="openai:gpt-3.5-turbo",
    system_prompt='You are the judge of a debate round. Evaluate the arguments. Consider this debate round from a technical standpoint, as to who had a better impact calculus and logic, and decide a winner.'
)
