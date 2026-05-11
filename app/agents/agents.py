# Agents 
# Multi-Agent workflow built with CrewAI.

# Agents:
#   1. Query Understanding Agent  -> intent + category
#   2. RAG Retrieval Agent        -> uses our RAG tool
#   3. Sentiment Analysis Agent   -> urgency / emotion
#   4. Escalation Agent           -> human handoff decision

# All four agents share the SAME RAG pipeline as a tool.

import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Reuse the RAG pipeline we built in rag_pipeline.py
from app.rag import rag_pipeline

load_dotenv()


# RAG exposed as a CrewAI Tool
# @tool turns a plain function into something agents can call.
# Any agent we attach this to will be able to retrieve context.

@tool("rag_search")
def rag_search(query: str) -> str:
    """
    Search the enterprise customer support knowledge base
    using hybrid (semantic + keyword) retrieval and return
    a grounded answer with the matched context.
    """
    result = rag_pipeline.generate_answer(query)
    # Return a short, readable string the agent can reason over
    sources_preview = "\n".join(
        [f"- [{s['metadata'].get('category','?')}] {s['content'][:120]}..."
         for s in result["sources"][:3]]
    )
    return f"ANSWER:\n{result['answer']}\n\nTOP SOURCES:\n{sources_preview}"


# Building the 4 agents

def build_agents():
    """Create and return the four collaborating agents."""

    llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 1. Understands what the customer actually wants
    query_agent = Agent(
        role="Query Understanding Specialist",
        goal="Identify the customer's intent and the support category from their query.",
        backstory=(
            "You are an expert at reading short customer messages "
            "and labeling them with a clear intent (refund, cancel, info, complaint, ...) "
            "and a category (billing, account, shipping, product, ...)."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm_name,
    )

    # 2. Actually fetches context from the knowledge base RAG
    retrieval_agent = Agent(
        role="Knowledge Base Retrieval Agent",
        goal="Use the rag_search tool to find the most relevant support context and draft a grounded answer.",
        backstory=(
            "You always rely on the rag_search tool. "
            "You never invent facts. If the tool result is insufficient, "
            "you say so clearly."
        ),
        tools=[rag_search],
        verbose=False,
        allow_delegation=False,
        llm=llm_name,
    )

    # 3. Looks at tone/urgency
    sentiment_agent = Agent(
        role="Customer Sentiment Analyst",
        goal="Detect the customer's emotional state and urgency level (low / medium / high).",
        backstory=(
            "You read short messages and detect frustration, anger, "
            "confusion or satisfaction. You output one line: "
            "'Sentiment: <label> | Urgency: <low|medium|high>'."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm_name,
    )

    # 4. Decides whether to escalate
    escalation_agent = Agent(
        role="Escalation Decision Agent",
        goal="Decide whether the issue should be escalated to a human support agent.",
        backstory=(
            "You combine the retrieval result and the sentiment analysis. "
            "If the RAG answer is uncertain OR sentiment is negative with high urgency, "
            "you escalate. Otherwise you do not."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm_name,
    )

    return query_agent, retrieval_agent, sentiment_agent, escalation_agent



# Build the tasks (one per agent) and run the Crew

def run_agentic_workflow(user_query: str) -> dict:
    """
    Run the full 4-agent pipeline on a customer query.
    Returns: final_response, retrieved_sources, escalation_flag, sentiment.
    """
    query_agent, retrieval_agent, sentiment_agent, escalation_agent = build_agents()

    # Task 1: classify the query
    task_understand = Task(
        description=(
            f"Customer query: '{user_query}'.\n"
            "Identify the intent and category. "
            "Return a single short line: 'Intent: <x> | Category: <y>'."
        ),
        expected_output="One line with intent and category.",
        agent=query_agent,
    )

    # Task 2: retrieve + draft a grounded answer
    task_retrieve = Task(
        description=(
            f"Customer query: '{user_query}'.\n"
            "Use the rag_search tool to fetch context and draft a clear, "
            "grounded answer for the customer. "
            "If the context does not contain the answer, say so."
        ),
        expected_output="A grounded customer-facing answer.",
        agent=retrieval_agent,
    )

    # Task 3: sentiment + urgency
    task_sentiment = Task(
        description=(
            f"Customer query: '{user_query}'.\n"
            "Output exactly one line: 'Sentiment: <positive|neutral|negative> "
            "| Urgency: <low|medium|high>'."
        ),
        expected_output="One line with sentiment and urgency.",
        agent=sentiment_agent,
    )

    #Task 4: escalation decision
    task_escalate = Task(
        description=(
            "Based on the retrieval result and the sentiment analysis, "
            "decide whether to escalate to a human. "
            "Reply with exactly one of: 'ESCALATE: YES' or 'ESCALATE: NO', "
            "followed by a 1-sentence reason."
        ),
        expected_output="ESCALATE: YES/NO + 1 sentence reason.",
        agent=escalation_agent,
        # This task depends on the earlier ones -> CrewAI passes their output as context
        context=[task_retrieve, task_sentiment],
    )

    # Assemble the Crew
    crew = Crew(
        agents=[query_agent, retrieval_agent, sentiment_agent, escalation_agent],
        tasks=[task_understand, task_retrieve, task_sentiment, task_escalate],
        process=Process.sequential,  # run tasks one after another(sequentially)
        verbose=False,
    )

    # Run the crew
    crew_output = crew.kickoff()

    # Also fetch raw RAG sources directly 
    rag_result = rag_pipeline.generate_answer(user_query)

    # Pull the per-task outputs out of CrewAI's result object
    # (each task_output.raw is a string)
    task_outputs = [t.raw for t in crew_output.tasks_output]

    final_answer = task_outputs[1] if len(task_outputs) > 1 else str(crew_output)
    sentiment_line = task_outputs[2] if len(task_outputs) > 2 else ""
    escalation_line = task_outputs[3] if len(task_outputs) > 3 else "ESCALATE: NO"

    # Simple flag parsing
    escalate_flag = "ESCALATE: YES" in escalation_line.upper()

    return {
        "final_response": final_answer,
        "retrieved_sources": rag_result["sources"],
        "sentiment": sentiment_line,
        "escalation": escalation_line,
        "escalate_flag": escalate_flag,
    }


# Quick local test

if __name__ == "__main__":
    # Make sure RAG is initialized first
    rag_pipeline.build_rag_pipeline()
    out = run_agentic_workflow("I want to cancel my order, I'm really frustrated!")
    print("\n--- FINAL ---")
    for k, v in out.items():
        if k != "retrieved_sources":
            print(f"{k}: {v}\n")
