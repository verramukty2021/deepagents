"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import tavily_search, think_tool

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Create research sub-agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}


model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)


# ------------------------------------------------------
# Agent Prompt = Research Workflow + Sub-Agent Research Coordination
# ------------------------------------------------------

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=3,
        max_researcher_iterations=1,
    )
)
# ------------------------------------------------------

# Create the agent
# recursion_limit = hard cap jumlah node execution di LangGraph.
# Kalkulasi worst case:
#   orchestrator steps (~15) + 3 sub-agent x 15 steps (~45) = ~60
# Set 100 memberi ruang aman tanpa risiko infinite loop.
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
).with_config({"recursion_limit": 200})
