# =============================================================================
# Mental Wellness Practice Suggester -- A LangGraph Learning Project
# =============================================================================
#
# This project teaches you how LangGraph works by building a mental wellness
# assistant that suggests personalized calming practices.
#
# WHAT THIS DOES:
# A user enters how they are feeling (e.g. "I feel stressed", "I can't sleep",
# "I feel anxious and overwhelmed"). The system runs 3 suggestion engines in
# PARALLEL (breathing, mindfulness, movement), then a decision node picks the
# best approach and routes to either a QUICK practice (under 5 minutes) or a
# DEEPER session (10-15 minutes) based on severity.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- user feeling flows through the graph
# 2. Nodes -- each function does one job (suggest breathing, mindfulness, etc.)
# 3. Parallel Execution -- 3 suggestion nodes run at the same time
# 4. Fan-in -- waiting for all 3 suggestions before picking the best
# 5. Conditional Edges -- routing to quick vs deep based on severity
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#   START
#     |
#   understand_mood
#     |
#     +---> suggest_breathing --------+
#     |                               |
#     +---> suggest_mindfulness ------+---> pick_best_practice
#     |                               |         |
#     +---> suggest_movement ---------+    (conditional)
#                                        /          \
#                                   quick?         deep?
#                                     |               |
#                               quick_practice   deep_practice
#                                     |               |
#                                    END             END
#
# HOW TO RUN:
#   python mental_wellness_graph.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

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


class WellnessState(BaseModel):
    user_feeling: str = ""
    breathing_suggestion: str = ""
    mindfulness_suggestion: str = ""
    movement_suggestion: str = ""
    movie_suggestion: str = ""
    needs_deep_session: bool = False
    practice_reason: str = ""
    final_suggestion: str = ""
    messages: Annotated[list, operator.add] = []
    #messages: list = []

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def understand_mood(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a compassionate mental wellness assistant. "
        f"A user says: '{state.user_feeling}'. "
        f"Acknowledge their feeling warmly in 1-2 sentences. "
        f"Then classify the severity as MILD, MODERATE, or HIGH in one word on a new line like: Severity: MILD"
    )
    return {
        "messages": [f"[understand_mood] {response.content}"]
    }


def suggest_breathing(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a breathing exercise specialist. "
        f"The user feels: '{state.user_feeling}'. "
        f"Suggest ONE specific breathing technique that would help. "
        f"Include the name, step-by-step instructions (3-4 steps), and how long it takes. "
        f"Keep it under 5 sentences."
    )
    return {
        "breathing_suggestion": response.content,
        "messages": [f"[suggest_breathing] Done"]
    }


def suggest_mindfulness(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a mindfulness and meditation guide. "
        f"The user feels: '{state.user_feeling}'. "
        f"Suggest ONE specific mindfulness or grounding exercise that would help. "
        f"Include the name, simple instructions, and duration. "
        f"Keep it under 5 sentences."
    )
    return {
        "mindfulness_suggestion": response.content,
        "messages": [f"[suggest_mindfulness] Done"]
    }


def suggest_movement(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a gentle movement and body wellness coach. "
        f"The user feels: '{state.user_feeling}'. "
        f"Suggest ONE specific gentle physical activity or stretch that would help. "
        f"Include the name, simple instructions, and duration. "
        f"Keep it under 5 sentences."
    )
    return {
        "movement_suggestion": response.content,
        "messages": [f"[suggest_movement] Done"]
    }


def pick_best_practice(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a wellness decision system. The user feels: '{state.user_feeling}'.\n\n"
        f"Here are three suggestions from specialists:\n\n"
        f"BREATHING:\n{state.breathing_suggestion}\n\n"
        f"MINDFULNESS:\n{state.mindfulness_suggestion}\n\n"
        f"MOVEMENT:\n{state.movement_suggestion}\n\n"
        f"Decide: does this person need a QUICK practice (under 5 min, for mild/moderate feelings) "
        f"or a DEEP session (10-15 min, for high stress/anxiety/overwhelm)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_deep_session": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        needs_deep = result["needs_deep_session"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep = False
        reason = "Could not parse decision, defaulting to quick practice."

    return {
        "needs_deep_session": needs_deep,
        "practice_reason": reason,
        "messages": [f"[pick_best_practice] deep_session={needs_deep}"]
    }


def quick_practice(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a friendly wellness coach. The user feels: '{state.user_feeling}'.\n\n"
        f"Based on these specialist suggestions, create a SHORT practice (under 5 minutes) "
        f"that combines the best elements:\n\n"
        f"BREATHING: {state.breathing_suggestion}\n"
        f"MINDFULNESS: {state.mindfulness_suggestion}\n"
        f"MOVEMENT: {state.movement_suggestion}\n\n"
        f"Format it as a simple numbered list of steps. "
        f"Keep it warm, encouraging, and easy to follow. End with a kind closing line."
    )
    return {
        "final_suggestion": f"QUICK WELLNESS PRACTICE (under 5 min)\n{'='*45}\n{response.content}",
        "messages": [f"[quick_practice] Generated quick practice"]
    }


def deep_practice(state: WellnessState) -> dict:
    response = llm.invoke(
        f"You are a compassionate wellness coach. The user feels: '{state.user_feeling}'.\n\n"
        f"Based on these specialist suggestions, create a DEEPER session (10-15 minutes) "
        f"that thoughtfully combines all three approaches:\n\n"
        f"BREATHING: {state.breathing_suggestion}\n"
        f"MINDFULNESS: {state.mindfulness_suggestion}\n"
        f"MOVEMENT: {state.movement_suggestion}\n\n"
        f"Structure it in 3 phases: Settle (breathing), Ground (mindfulness), Release (movement). "
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it warm and supportive. End with a kind closing message."
    )
    return {
        "final_suggestion": f"DEEP WELLNESS SESSION (10-15 min)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_practice] Generated deep session"]
    }


def route_after_decision(state: WellnessState) -> str:
    if state.needs_deep_session:
        return "deep"
    else:
        return "quick"


graph = StateGraph(WellnessState)

graph.add_node("understand_mood", understand_mood)
graph.add_node("suggest_breathing", suggest_breathing)
graph.add_node("suggest_mindfulness", suggest_mindfulness)
graph.add_node("suggest_movement", suggest_movement)
graph.add_node("pick_best_practice", pick_best_practice)
graph.add_node("quick_practice", quick_practice)
graph.add_node("deep_practice", deep_practice)

graph.add_edge(START, "understand_mood")

graph.add_edge("understand_mood", "suggest_breathing")
graph.add_edge("understand_mood", "suggest_mindfulness")
graph.add_edge("understand_mood", "suggest_movement")

graph.add_edge("suggest_breathing", "pick_best_practice")
graph.add_edge("suggest_mindfulness", "pick_best_practice")
graph.add_edge("suggest_movement", "pick_best_practice")

graph.add_conditional_edges(
    "pick_best_practice",
    route_after_decision,
    {
        "quick": "quick_practice",
        "deep": "deep_practice",
    }
)

graph.add_edge("quick_practice", END)
graph.add_edge("deep_practice", END)

app = graph.compile()


def run_wellness_check(feeling: str):
    print("=" * 55)
    print("  MENTAL WELLNESS PRACTICE SUGGESTER")
    print(f"  You said: \"{feeling}\"")
    print("=" * 55)

    result = app.invoke({
        "user_feeling": feeling,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR PERSONALIZED PRACTICE")
    print("=" * 55)
    print(f"\n{result['final_suggestion']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MENTAL WELLNESS PRACTICE SUGGESTER")
    print("=" * 55)
    print("\n  Tell me how you're feeling and I'll suggest a")
    print("  personalized wellness practice just for you.")
    print("  Type 'quit' to exit.\n")

    while True:
        feeling = input("  How are you feeling? > ").strip()

        if feeling.lower() in ("quit", "exit", "q"):
            print("\n  Take care of yourself. Goodbye!\n")
            break

        if not feeling:
            continue

        run_wellness_check(feeling)
        print("\n")
