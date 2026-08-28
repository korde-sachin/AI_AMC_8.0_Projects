#              [ user pastes code ]
#                       |
#                       v
#            +----------------------+
#            |   understand_code    |   <- classifier: acknowledges the code,
#            |                      |      identifies language/purpose
#            +----------+-----------+
#                       |
#     PARALLEL FAN-OUT (three specialists read the same code
#     independently — none of them needs to wait for another)
#      /                |                 \
#     v                 v                  v
#+----------+   +-----------------+   +----------------+
#| explain_ |   | identify_code_  |   | add_code_      |
#| code_    |   |    risks        |   |   comments     |
#| plain_   |   |                 |   |                |
#| english  |   |                 |   |                |
#+----+-----+   +--------+--------+   +--------+-------+
#     \                  |                     /
#      FAN-IN (waits for all three to finish)
#                        |
#                        v
#            +-----------------------+
#            |    decide_review_     |   <- reads all 3 outputs,
#            |        depth          |      decides simple vs deep
#            +-----------+-----------+
#                         |
#                CONDITIONAL EDGE
#                /                 \
#        simple                     deep
#          |                          |
#+---------+---------+     +---------+----------+
#| simple_code_       |     | deep_code_review   |
#| explanation         |     |                     |
#+---------+---------+     +---------+----------+
#                \                  /
#                 v                v
#                     [ END ]
#*/
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

# STATE — the shared object every node in this graph reads from and writes
# to. Nodes never call each other directly; the only way information moves
# from one node to another is through this object.
class CodeExplainerState(BaseModel):
    user_code: str = ""                     # input — the code the user pasted in
    code_context: str = ""                  # written by understand_code (language/purpose guess)
    plain_english_explanation: str = ""     # written by explain_code_plain_english
    code_risks: str = ""                    # written by identify_code_risks
    commented_code: str = ""                # written by add_code_comments
    needs_deep_review: bool = False         # written by decide_review_depth — the actual decision
    review_decision_reason: str = ""        # written alongside it — the "why" behind the decision
    final_output: str = ""                  # written by whichever final node runs (simple or deep)
    messages: Annotated[list, operator.add] = []
    # ^ REDUCER FIELD — every other field above is last-write-wins: whoever
    # writes it last, that's the value that sticks. This one is different.
    # operator.add tells LangGraph to CONCATENATE every write instead of
    # overwriting — required because explain_code_plain_english,
    # identify_code_risks, and add_code_comments all run in PARALLEL and
    # all write to "messages" in the same step. Without this, that would
    # raise InvalidUpdateError — the same error you already triggered
    # deliberately on the wellness graph.

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# NODE — the classifier. Lightweight on purpose: just enough to tell the
# three specialists what they're looking at, not to do their job for them.
def understand_code(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a technical code reviewer. "
        f"The user provided this code: '{state.user_code}'. "
        f"In 1-2 sentences, identify the programming language and briefly "
        f"describe what this code is generally trying to do. "
        f"Do not explain it in detail — that happens later."
    )
    return {
        "code_context": response.content,
        "messages": [f"[understand_code] Done"]
    }


# NODE — specialist 1 of 3. Runs in parallel with identify_code_risks and
# add_code_comments, once understand_code finishes.
def explain_code_plain_english(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a technical code reviewer. "
        f"The user provided this code: '{state.user_code}'. "
        f"In 4-5 sentences, explain code in plain english and what it's objective. "
        f"Do not explain any other aspect of the code."
    )
    return {
        "plain_english_explanation": response.content,
        "messages": [f"[explain_code_plain_english] Done"]
    }

# NODE — specialist 2 of 3. Runs in parallel with explain_code_plain_english and
# add_code_comments, once understand_code finishes.
def identify_code_risks(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a technical code reviewer. "
        f"The user provided this code: '{state.user_code}'. "
        f"In 4-5 sentences, identify bugs, edge cases, or unclear parts in this code. "
        f"Do not explain any other aspect of the code."
    )
    return {
        "code_risks": response.content,
        "messages": [f"[identify_code_risks] Done"]
    }
# NODE — specialist 3 of 3. Runs in parallel with explain_code_plain_english and
# identify_code_risks, once understand_code finishes.
def add_code_comments(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a technical code reviewer. "
        f"The user provided this code: '{state.user_code}'. "
        f"Return the complete code again, unchanged in logic, but with clear "
        f"inline comments added wherever they would help a reader understand it. "
        f"Do not change the code's functionality — only add comments."
    )
    return {
        "commented_code": response.content,
        "messages": [f"[add_code_comments] Done"]
    }

# NODE — the decision point. Waits for all three specialists.
def decide_review_depth(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a senior code reviewer making a triage call.\n\n"
        f"CODE CONTEXT: {state.code_context}\n\n"
        f"PLAIN ENGLISH EXPLANATION:\n{state.plain_english_explanation}\n\n"
        f"IDENTIFIED RISKS:\n{state.code_risks}\n\n"
        f"Decide: is this code simple enough that a short plain-English explanation "
        f"is sufficient (SIMPLE), or do the identified risks mean it needs a deeper, "
        f"more thorough review (DEEP)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_deep_review": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        needs_deep = result["needs_deep_review"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep = True
        reason = "Could not parse decision, defaulting to deep review for safety."

    return {
        "needs_deep_review": needs_deep,
        "review_decision_reason": reason,
        "messages": [f"[decide_review_depth] deep={needs_deep}"]
    }

# ROUTER
def route_after_review(state: CodeExplainerState) -> str:
    if state.needs_deep_review:
        return "deep"
    return "simple"


# NODE — final output when SIMPLE.
def simple_code_explanation(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a code reviewer producing a final summary for a user.\n\n"
        f"EXPLANATION: {state.plain_english_explanation}\n\n"
        f"Write a short, friendly explanation of what this code does, suitable "
        f"for someone who just wants to understand it quickly. Keep it concise."
    )
    return {
        "final_output": response.content,
        "messages": [f"[simple_code_explanation] Done"]
    }


# NODE — final output when DEEP.
def deep_code_review(state: CodeExplainerState) -> dict:
    response = llm.invoke(
        f"You are a senior code reviewer producing a full review report.\n\n"
        f"EXPLANATION: {state.plain_english_explanation}\n\n"
        f"RISKS: {state.code_risks}\n\n"
        f"COMMENTED CODE:\n{state.commented_code}\n\n"
        f"REASON FOR DEEP REVIEW: {state.review_decision_reason}\n\n"
        f"Write a structured review report covering: what the code does, the risks found, "
        f"and the commented version of the code. This should be thorough."
    )
    return {
        "final_output": response.content,
        "messages": [f"[deep_code_review] Done"]
    }

graph = StateGraph(CodeExplainerState)


graph.add_node("understand_code", understand_code)
graph.add_node("explain_code_plain_english", explain_code_plain_english)
graph.add_node("identify_code_risks", identify_code_risks)
graph.add_node("add_code_comments", add_code_comments)
graph.add_node("decide_review_depth", decide_review_depth)
#graph.add_node("route_after_review", route_after_review)
graph.add_node("simple_code_explanation", simple_code_explanation)
graph.add_node("deep_code_review", deep_code_review)

graph.add_edge(START, "understand_code")

graph.add_edge("understand_code", "explain_code_plain_english")
graph.add_edge("understand_code", "identify_code_risks")
graph.add_edge("understand_code", "add_code_comments")

graph.add_edge("explain_code_plain_english", "decide_review_depth")
graph.add_edge("identify_code_risks", "decide_review_depth")
graph.add_edge("add_code_comments", "decide_review_depth")

graph.add_conditional_edges(
    "decide_review_depth",
    route_after_review,
    {
        "deep": "deep_code_review",
        "simple": "simple_code_explanation",
    }
)

graph.add_edge("deep_code_review", END)
graph.add_edge("simple_code_explanation", END)

app = graph.compile()

def run_code_explainer(code: str):
    print("=" * 55)
    print("  CODE EXPLAINER")
    print("=" * 55)
    preview = code if len(code) <= 200 else code[:200] + " ...(truncated)"
    print(f"  You pasted:\n{preview}")
    print("=" * 55)

    result = app.invoke({
        "user_code": code,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  REVIEW RESULT")
    print("=" * 55)
    print(f"\n{result['final_output']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  CODE EXPLAINER")
    print("=" * 55)
    print("\n  Paste your code below. When you're done, type END on")
    print("  its own line and press Enter.")
    print("  Type 'quit' instead of pasting code to exit.\n")

    while True:
        first_line = input("  Paste code (or 'quit') > ")

        if first_line.strip().lower() in ("quit", "exit", "q"):
            print("\n  Happy coding. Goodbye!\n")
            break

        lines = [first_line]
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)

        code = "\n".join(lines).strip()

        if not code:
            continue

        run_code_explainer(code)
        print("\n")



