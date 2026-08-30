"""Blog Post Writer Agent — a LangChain agent that write a blog from an idea.

Setup: pip install -r requirements.txt, copy .env.example to .env, add your key.
Run:   python blog_post_writer_agent.py
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("email_humanizer")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

OUTLINE_PROMPT = PromptTemplate(
    input_variables=["topic","target_audience","keywords"],
    template="""You are a professional content strategist and SEO copywriter.
Create a blog outline for the following:
 
Topic: {topic}
Target audience: {target_audience}
Target keywords (weave these in naturally, don't force them): {keywords}
 
Produce:
- 3 attention-grabbing title options (choose keyword-friendly phrasing without keyword-stuffing)
- An introduction angle: the hook/angle the intro should open with
- Section headings as an H2/H3 hierarchy that logically develops the topic
- A conclusion section that includes a clear call-to-action
 
Note briefly, next to the relevant heading, where each target keyword should
naturally appear (e.g. title, an H2, or the intro).
 
Return the complete outline as structured Markdown (title options, then
headings), not prose.""",
)

BLOG_POST_PROMPT = PromptTemplate(
    input_variables=["outline"],
    template="""You are a professional content writer. Expand the following
blog outline into a complete, polished blog post.
 
Outline:
{outline}
 
Rules:
- Pick the strongest title option from the outline (or refine it) and use it as the post's H1
- Open with a genuine hook — a question, a surprising fact, or a relatable scenario — not a
  generic restatement of the topic
- Follow the outline's H2/H3 structure, but write full, smoothly-transitioning prose under
  each heading — don't just restate the headings as bullet points
- Weave the target keywords in naturally wherever the outline notes them; never stuff them
- Target 600-800 words total
- End with a clear, specific call-to-action
 
Return ONLY the finished blog post in Markdown.""",
)


@tool  # this decorator registers the function below as a callable "tool" the agent can invoke by name
def create_blog_outline(topic: str, target_audience: str, keywords: str) -> str:
    """Create an SEO-friendly blog outline from a topic, target audience, and target keywords.
 
    Use this FIRST, before write_blog_post. `keywords` should be the 2-3 keywords the
    post should rank for (comma-separated is fine).
    """
    logger.info(
        "[create_blog_outline] topic=%r audience=%r keywords=%r",
        topic, target_audience, keywords,
    )
 
    return str(
        llm.invoke(
            OUTLINE_PROMPT.format(
                topic=topic, target_audience=target_audience, keywords=keywords
            )
        ).content
    )

 
@tool
def write_blog_post(outline: str) -> str:
    """Expand a blog outline into a complete 600-800 word blog post in Markdown.
    Use this AFTER create_blog_outline, passing in the outline it produced.
    """
    # This docstring tells the agent this tool should run AFTER create_blog_outline —
    # it takes the structured outline (not the raw user text) as input.
    logger.info("[write_blog_post] expanding outline (%d chars)", len(outline))
    return str(llm.invoke(BLOG_POST_PROMPT.format(outline=outline)).content)

# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional content writer who turns any topic into an
engaging, SEO-aware blog post.
 
When the user gives you a blog topic (with a target audience and target keywords), follow
these steps in order:
1. First, call the create_blog_outline tool to structure the post: title options, an
   introduction angle, H2/H3 section headings, and a conclusion with a call-to-action —
   weaving the target keywords in naturally.
2. Then, call the write_blog_post tool, passing it the exact outline produced in step 1, to
   expand it into a complete 600-800 word blog post with a hook opening, smooth transitions,
   and a closing call-to-action.
3. Return the finished blog post to the user in Markdown. Do not add commentary before or
   after it.
 
Always use both tools, in this order: create_blog_outline, then write_blog_post. Never write
the post yourself without first producing an outline through create_blog_outline"""

agent = create_agent(model=llm, tools=[create_blog_outline, write_blog_post], system_prompt=SYSTEM_PROMPT)


def run_blog_writer_agent(request: str) -> str:
    """Run the agent on a raw blog request and return the finished Markdown blog post."""
    result = agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _prompt_for_request() -> str:
    """Collect topic, audience, and keywords as separate prompts, then combine them into
    one request string for the agent. Asking for them separately (rather than one free-text
    line) makes it far more reliable for the LLM to extract clean arguments for
    create_blog_outline's three parameters."""
    topic = input("Blog topic: ").strip()
    audience = input("Target audience: ").strip()
    keywords = input("Target keywords (2-3, comma-separated): ").strip()
    return (
        f"Write a blog post.\nTopic: {topic}\nTarget audience: {audience}\n"
        f"Target keywords: {keywords}"
    )
 
 
def main() -> None:
    # Simple command-line loop: greet the user and explain how to exit.
    print("\nBlog Writer Agent (LangChain + OpenAI)")
    print("Describe the blog post you want. Type 'quit' to exit.\n")
 
    while True:
        topic_check = input("Start a new blog post? (Enter to continue, 'quit' to exit): ").strip()
        if topic_check.lower() in ("quit", "exit", "q"):
            break
 
        request = _prompt_for_request()
        if not request.strip():
            continue
 
        try:
            # Run the full two-tool agent pipeline on the collected request.
            blog_post = run_blog_writer_agent(request)
            # Pretty-print the result between two lines of "=" for readability in the terminal.
            print("\n" + "=" * 60)
            print(blog_post)
            print("=" * 60 + "\n")
        except Exception as e:
            # Catch-all so one failed request (e.g. an API error) doesn't crash the whole CLI loop —
            # log it and let the user try again.
            logger.error("Agent failed: %s", e)
 
if __name__ == "__main__":
    main()