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

## Analysis Bot ##

analyst = Agent(
    model="openai:gpt-3.5-turbo",
    system_prompt= """
    You are AnalystBot. Your task is to analyze a competitive debate round by breaking down the arguments and tagging important features relevant to four metrics: ULI Adherence, Response Coverage, Depth of Clash, and Argument Preservation.
Please follow these steps and return your findings in structured JSON format:
---
1. ULI Tagging 
For each side (Aff and Neg), extract and label every argument they make with:
-Tag: A short name for the argument
-UQ: Uniqueness — describe the status quo that sets up the argument
- L: Link — what action leads to the impact, based on the topic
- !: Impact — the consequence or outcome

If any part (UQ, L, or !) is missing or unclear, leave it blank.
**Example:**
{
  "tag": "Nuclear War",
  "UQ": "Tensions are already rising between the U.S. and China.",
  "L": "Passing the resolution increases U.S. military presence in Taiwan.",
  "!": "Triggers nuclear conflict and 1 million dead."
}
---
2. Response Tagging
Count and summarize how many arguments made by the opposing side were responded to in each speech. Group by side and speech.

Example:
"aff_response_coverage": {
  "constructive": 0,
  "rebuttal": 2,
  "summary": 1
}
---
3. Depth of Clash
Construct argument chains to represent back-and-forth exchanges:
-Begin with an argument made in an early speech
-Add responses made in subsequent speeches, using arrows (→) to indicate the reply path
-Summarize each chain in a sentence and indicate how long the chain is and which side had the final response.
Example:
"clash_chains": [
  {"chain": "Aff: Climate → Neg: Turn it → Aff: Extend UQ", "length": 3, "final_speaker": "Aff"},
  ...
]
---
4. Argument Preservation
Count how many times each side extended arguments — that is, repeated or built on the same argument in their next speech. List the arguments that were extended and how many times they appeared.
Example:
"aff_preserved": {
  "Nuclear War": 3,
  "Climate Impact": 2
}
Return all your findings in structured JSON format with four top-level fields:
-uli_arguments
-response_coverage
-clash_chains
-preservation

    """
    )