import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


# STATE — the shared object every node reads from and writes to.
# Nodes never call each other directly; they only communicate through this.
class IncidentState(BaseModel):
    incident_details: str = ""              # input — set once, by the caller
    incident_urgency: str = ""              # written by understand_incident
    incident_root_cause: str = ""           # written by diagnose_root_cause
    incident_impact: str = ""               # written by diagnose_impact
    incident_first_remedy: str = ""         # written by suggest_first_remedy
    incident_routine_issue: bool = False    # written later, by the decision node
    incident_short_routine_fix: str = ""    # written later, by a final node
    incident_escalation_summary: str = ""   # written later, by the other final node
    incident_decision_reason: str = ""      # written later, alongside the decision
    messages: Annotated[list, operator.add] = []
    # ^ REDUCER — plain fields are last-write-wins; this one is different.
    # operator.add tells LangGraph to CONCATENATE multiple writes instead of
    # overwriting, which is required because several nodes below run in
    # PARALLEL and all write to "messages" in the same step.


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
# temperature=0.3, not 0.7 like the wellness project — this LLM is being used
# for classification (urgency, root cause), where consistency matters more
# than creative variation.


# NODE — the classifier. First node to run. Reads incident_details, writes
# incident_urgency. Every node below has this exact same shape: read state,
# make one focused LLM call, return a dict of updates.
def understand_incident(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT support triage assistant. "
        f"An incident has been reported: '{state.incident_details}'. "
        f"Briefly acknowledge the incident in 1-2 sentences. "
        f"Then, on a new line, write exactly: Urgency: HIGH  (or MEDIUM, or LOW)"
    )

    last_line = response.content.strip().split("\n")[-1]
    urgency = last_line.split(":")[-1].strip()

    return {
        "incident_urgency": urgency,
        "messages": [f"[understand_incident] {response.content}"]
    }


# NODE — specialist 1 of 3. Runs in PARALLEL with the next two, once
# understand_incident finishes (that's decided later, by the edges — not
# by anything in this function itself).
def diagnose_root_cause(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT diagnostics specialist. "
        f"An incident has been reported: '{state.incident_details}'. "
        f"Suggest the most likely root cause of this issue in 2-3 sentences. "
        f"Be specific about what component or system is likely responsible."
    )
    return {
        "incident_root_cause": response.content,
        "messages": [f"[diagnose_root_cause] Done"]
    }


# NODE — specialist 2 of 3.
def diagnose_impact(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT diagnostics specialist. "
        f"An incident has been reported: '{state.incident_details}'. "
        f"Diagnose its impact — which systems or users are likely affected and how severely."
    )
    return {
        "incident_impact": response.content,
        "messages": [f"[diagnose_impact] Done"]
    }


# NODE — specialist 3 of 3.
def suggest_first_remedy(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT diagnostics specialist. "
        f"An incident has been reported: '{state.incident_details}'. "
        f"Give one or two immediate first steps someone could try right now."
    )
    return {
        "incident_first_remedy": response.content,
        "messages": [f"[suggest_first_remedy] Done"]
    }

# NODE — the decision point. Waits for all 3 specialists (FAN-IN happens
# here, via the edges we wire next — this function itself doesn't know or
# care that it's waiting on three parallel branches).
def make_decision(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT incident triage lead. An incident was reported: '{state.incident_details}'.\n\n"
        f"Here is what your specialists found:\n\n"
        f"ROOT CAUSE:\n{state.incident_root_cause}\n\n"
        f"IMPACT:\n{state.incident_impact}\n\n"
        f"FIRST REMEDY:\n{state.incident_first_remedy}\n\n"
        f"Urgency was classified as: {state.incident_urgency}\n\n"
        f"Decide: can first-line support resolve this with the suggested remedy (ROUTINE), "
        f"or does it need an on-call engineer (ESCALATE)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"is_routine": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        is_routine = result["is_routine"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        is_routine = False
        reason = "Could not parse decision, defaulting to escalation for safety."

    return {
        "incident_routine_issue": is_routine,
        "incident_decision_reason": reason,
        "messages": [f"[make_decision] routine={is_routine}"]
    }


# ROUTER — not a node, just a plain function that reads state and returns a
# label. Paired with add_conditional_edges, not add_edge.
def route_after_decision(state: IncidentState) -> str:
    if state.incident_routine_issue:
        return "routine"
    return "escalate"

# NODE — final output when the decision was ROUTINE.
def generate_routine_fix(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT support engineer. An incident was reported: '{state.incident_details}'.\n\n"
        f"Based on this diagnosis:\n"
        f"ROOT CAUSE: {state.incident_root_cause}\n"
        f"FIRST REMEDY: {state.incident_first_remedy}\n\n"
        f"Write a short, step-by-step runbook a first-line support person can follow to resolve this. "
        f"Keep it under 6 steps."
    )
    return {
        "incident_short_routine_fix": response.content,
        "messages": [f"[generate_routine_fix] Done"]
    }


# NODE — final output when the decision was ESCALATE.
def generate_escalation_summary(state: IncidentState) -> dict:
    response = llm.invoke(
        f"You are an IT incident commander preparing a handoff to an on-call engineer.\n\n"
        f"INCIDENT: {state.incident_details}\n"
        f"URGENCY: {state.incident_urgency}\n"
        f"ROOT CAUSE: {state.incident_root_cause}\n"
        f"IMPACT: {state.incident_impact}\n"
        f"FIRST REMEDY ATTEMPTED/SUGGESTED: {state.incident_first_remedy}\n"
        f"WHY ESCALATED: {state.incident_decision_reason}\n\n"
        f"Write a concise escalation summary the engineer can act on immediately, "
        f"including what's known and what still needs investigation."
    )
    return {
        "incident_escalation_summary": response.content,
        "messages": [f"[generate_escalation_summary] Done"]
    }

# 1. Create the container
graph = StateGraph(IncidentState)

# 2. Register all six nodes
graph.add_node("understand_incident", understand_incident)
graph.add_node("diagnose_root_cause", diagnose_root_cause)
graph.add_node("diagnose_impact", diagnose_impact)
graph.add_node("suggest_first_remedy", suggest_first_remedy)
graph.add_node("make_decision", make_decision)
graph.add_node("generate_routine_fix", generate_routine_fix)
graph.add_node("generate_escalation_summary", generate_escalation_summary)

# 3. START -> classifier
graph.add_edge(START, "understand_incident")

# 4. Parallel fan-out: three edges, same source
graph.add_edge("understand_incident", "diagnose_root_cause")
graph.add_edge("understand_incident", "diagnose_impact")
graph.add_edge("understand_incident", "suggest_first_remedy")

# 5. Fan-in: three edges, same destination
graph.add_edge("diagnose_root_cause", "make_decision")
graph.add_edge("diagnose_impact", "make_decision")
graph.add_edge("suggest_first_remedy", "make_decision")

# 6. Conditional edge: one function decides between two named destinations
graph.add_conditional_edges(
    "make_decision",
    route_after_decision,
    {
        "routine": "generate_routine_fix",
        "escalate": "generate_escalation_summary",
    }
)

# 7. Both final nodes -> END
graph.add_edge("generate_routine_fix", END)
graph.add_edge("generate_escalation_summary", END)

# 8. Builder -> runnable
app = graph.compile()


def run_incident_triage(details: str):
    print("=" * 55)
    print("  IT INCIDENT TRIAGE ASSISTANT")
    print(f"  Incident: \"{details}\"")
    print("=" * 55)

    result = app.invoke({
        "incident_details": details,
        "messages": [],
    })

    print("\n" + "=" * 55)
    if result["incident_routine_issue"]:
        print("  ROUTINE FIX")
        print("=" * 55)
        print(f"\n{result['incident_short_routine_fix']}")
    else:
        print("  ESCALATION REQUIRED")
        print("=" * 55)
        print(f"\n{result['incident_escalation_summary']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    run_incident_triage("Users can't log into the CRM since 9am, getting a timeout error.")

# No graph wiring yet — StateGraph, add_node, add_edge, compile() all come
# next, once every node is confirmed working on its own.
#if __name__ == "__main__":
#    test_state = IncidentState(incident_details="Users can't log into the CRM since 9am, getting a timeout error.")
#    print(understand_incident(test_state))
#    print(diagnose_root_cause(test_state))
#    print(diagnose_impact(test_state))
#    print(suggest_first_remedy(test_state))
#
#    decision_input = IncidentState(
#        incident_details="Users can't log into the CRM since 9am, getting a timeout error.",
#        incident_root_cause="Likely an authentication server overload.",
#        incident_impact="All users blocked from CRM access during business hours.",
#        incident_first_remedy="Restart the authentication service and check server load."
#    )
#    print(make_decision(decision_input))