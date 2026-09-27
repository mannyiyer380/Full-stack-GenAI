# LangGraph — Study Notes

> **Living document.** This file grows as new classes are covered. Each section is tagged with the
> class/notebook it came from so you can trace a concept back to its source material.
>
> Source notebooks covered so far (in order):
> 1. `Class-41-23-Aug-2026-Langgraph-Introduction/Langraph_introduction.ipynb`
> 2. `Class-42-29-Aug-2026-tool-calling-in-Agentic-AI/ReAct_agent_with_tool_calling.ipynb`
> 3. `Class-42-29-Aug-2026-tool-calling-in-Agentic-AI/tool_engineering.ipynb`
> 4. `Class-43-05-Sep-2026-Pydantic-langgraph-Memory/pydantic.ipynb`
> 5. `Class-43-05-Sep-2026-Pydantic-langgraph-Memory/complete_pydantic_0_to_204_all_code.ipynb` (reference)
> 6. `Class-43-05-Sep-2026-Pydantic-langgraph-Memory/memory_in_langgraph.ipynb`
> 7. `Class-44-06-Sep-2026-memory-managment/memory_managment.ipynb`
> 8. `Class-45-12-sep-2026-memory_managnment_extended_human_in_loop/memory_managment.ipynb`
> 9. `Class-46-13-sep-2026-human_in_loop/Human-in-Loop.ipynb`
> 10. `Class-47-19-sep-sub-graphs-different-workflow-orchestrator/subgraph.ipynb`
> 11. `Class-49-50-26th-27thSep-2026-complete-Multiagentic-flow/multiagentic-flow.ipynb`

---

## Table of Contents

1. [Why LangGraph? The Core Mental Model](#1-why-langgraph-the-core-mental-model)
2. [State — the shared memory of a graph](#2-state--the-shared-memory-of-a-graph)
3. [Building your first graph](#3-building-your-first-graph)
4. [MessagesState and chat models](#4-messagesstate-and-chat-models)
5. [Multi-node workflows and reading token usage](#5-multi-node-workflows-and-reading-token-usage)
6. [A full agentic RAG router (Supervisor → Validator loop)](#6-a-full-agentic-rag-router-supervisor--validator-loop)
7. [Tool Calling — concepts](#7-tool-calling--concepts)
8. [Node vs Tool — the decision that matters most](#8-node-vs-tool--the-decision-that-matters-most)
9. [Tool Engineering — every way to build a tool](#9-tool-engineering--every-way-to-build-a-tool)
10. [MCP (Model Context Protocol) tools](#10-mcp-model-context-protocol-tools)
11. [The ReAct Agent pattern](#11-the-react-agent-pattern)
12. [Alternative: Planner–Executor pattern (no `bind_tools`)](#12-alternative-plannerexecutor-pattern-no-bind_tools)
13. [Pydantic Fundamentals](#13-pydantic-fundamentals)
14. [Memory in LangGraph with `Runtime[Context]`](#14-memory-in-langgraph-with-runtimecontext)
15. [Memory Management — overview](#15-memory-management--overview)
16. [Short-term memory: sliding window](#16-short-term-memory-sliding-window)
17. [Short-term memory: summarize + recent messages](#17-short-term-memory-summarize--recent-messages)
18. [Long-term memory: vector-store RAG + conversation memory](#18-long-term-memory-vector-store-rag--conversation-memory)
19. [Production memory in LangGraph: Checkpointer + Store + langmem](#19-production-memory-in-langgraph-checkpointer--store--langmem)
20. [Persistent memory with SQLite](#20-persistent-memory-with-sqlite)
21. [Human-in-the-Loop (HITL)](#21-human-in-the-loop-hitl)
22. [Subgraphs](#22-subgraphs)
23. [Multi-Agent Patterns](#23-multi-agent-patterns)
24. [Quick-reference cheat sheet](#24-quick-reference-cheat-sheet)

---

## 1. Why LangGraph? The Core Mental Model

*(Class 41)*

A plain LLM call is **stateless**: every `llm.invoke(...)` is independent, and the model has no memory
of anything you didn't explicitly pass in. Real applications need more than one LLM call — they need
**steps that run in a sequence, sometimes conditionally, sometimes in a loop**, all while sharing data.

LangGraph models an application as a **graph**:

- **Node** — a Python function that does one unit of work (call an LLM, call a tool, transform data, etc.)
- **Edge** — a connection saying "after this node, go to that node"
- **State** — a shared dictionary-like object that flows through every node
- **START / END** — special sentinel markers for where the graph begins and finishes

```
START → node_1 → node_2 → ... → END
```

Every node receives the *current state*, and returns a **partial update** (not the full state) —
LangGraph merges that partial update back into the shared state automatically.

```python
def hello_node(input: str):
    return {"message": f"Hello, {input}!"}
```

This is the simplest possible "node" — just a function. It becomes a graph node once it's registered
with `add_node`.

---

## 2. State — the shared memory of a graph

*(Class 41)*

State is the contract that every node reads from and writes to. LangGraph supports several ways to
define it:

```
State Schema
│
├── TypedDict           class State(TypedDict): text: str
├── Dataclass            @dataclass class State: text: str
├── Pydantic              class State(BaseModel): text: str
└── Built-in MessagesState   messages: Annotated[list[AnyMessage], add_messages]
```

`TypedDict` is the most common choice for teaching/production because it's lightweight and just a
typed dictionary — no runtime validation overhead.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    message: str
```

**Important nuance:** a node's return value is *merged* into state, not replacing it wholesale. If a
node returns `{"message": "..."}`, only the `message` key changes; every other key in state stays as
it was. This is why you can have many nodes each touching a different slice of state.

Fields that a node might not always set should be marked optional:

```python
from typing_extensions import TypedDict, NotRequired

class State(TypedDict):
    question: str
    answer: NotRequired[str]          # not every node sets this
```

or, for a whole class of "not all keys required" state:

```python
class AgentState(TypedDict, total=False):
    question: str
    route: str
    ...
```

---

## 3. Building your first graph

*(Class 41)*

The recipe is always the same five steps:

```python
from langgraph.graph import StateGraph, START, END

# 1. Define state
class State(TypedDict):
    message: str

# 2. Define node function(s)
def hello_node(state: State):
    return {"message": f"Hello, {state['message']}!"}

# 3. Create the builder and register nodes
builder = StateGraph(State)
builder.add_node("first_hello_node", hello_node)

# 4. Wire edges (including START/END)
builder.add_edge(START, "first_hello_node")
builder.add_edge("first_hello_node", END)

# 5. Compile → get a runnable graph
graph = builder.compile()
```

Run it with `.invoke(...)`, passing an initial state dict:

```python
result = graph.invoke({"message": "from INDIA"})
result["message"]   # "Hello, from INDIA!"
```

Visualize any compiled graph as a diagram (very useful for debugging routing logic):

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Multi-node linear workflow** — nodes are chained by adding more edges:

```python
workflow = StateGraph(Manny)
workflow.add_node("fun1", function1)
workflow.add_node("fun2", function2)

workflow.add_edge(START, "fun1")
workflow.add_edge("fun1", "fun2")
workflow.add_edge("fun2", END)

app = workflow.compile()
```

Each node sees the *output* of the node before it, because state updates are merged in edge order.

---

## 4. MessagesState and chat models

*(Class 41)*

For chat-style apps, LangGraph ships a ready-made state shape: `MessagesState`. Internally it looks
like this:

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

The `add_messages` **reducer** is the key idea: instead of a node's returned `messages` list
*replacing* the stored list, LangGraph **appends** new messages onto the existing list. This is what
lets a chatbot's message history accumulate turn over turn, automatically.

```python
from langgraph.graph import MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.6, reasoning_format="hidden")

def call_model(state: MessagesState):
    messages = state["messages"]          # full conversation so far
    response = llm.invoke(messages)
    return {"messages": [response]}       # add_messages appends this, doesn't overwrite

workflow = StateGraph(MessagesState)
workflow.add_node("mybot", call_model)
workflow.add_edge(START, "mybot")
workflow.add_edge("mybot", END)

app = workflow.compile()

result = app.invoke({"messages": [HumanMessage(content="Hi hello, how are you?")]})
```

---

## 5. Multi-node workflows and reading token usage

*(Class 41)*

A realistic pattern: one node calls the LLM, a second node inspects metadata about that call (token
usage, cost, etc.) and stores it in state.

```python
class State(TypedDict):
    question: str
    answer: NotRequired[str]
    response: NotRequired[AIMessage]     # keep the full AIMessage for inspection
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    total_tokens: NotRequired[int]

def llm_node(state: State):
    response = llm.invoke(state["question"])
    # we don't count tokens ourselves — the AIMessage carries usage_metadata
    return {"answer": response.content, "response": response}

def token_counter(state: State):
    usage = state["response"].usage_metadata or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

workflow = StateGraph(State)
workflow.add_node("llm", llm_node)
workflow.add_node("token_counter", token_counter)
workflow.add_edge(START, "llm")
workflow.add_edge("llm", "token_counter")
workflow.add_edge("token_counter", END)
```

**Takeaway:** don't reinvent token counting — LangChain's `AIMessage.usage_metadata` already carries
`input_tokens`, `output_tokens`, `total_tokens` populated by the provider's response.

---

## 6. A full agentic RAG router (Supervisor → Validator loop)

*(Class 41 — "5th and advanced workflow")*

This is the class's capstone example and demonstrates several concepts at once:
- **Structured output** to force the LLM to return a typed decision
- **Conditional routing** (`add_conditional_edges`) to pick between multiple candidate nodes
- **A retry loop**: a validator can send the flow back to the supervisor with feedback
- Combining **RAG**, a **plain LLM answer**, and **live web search** as interchangeable "routes"

### 6.1 Structured routing decision

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    route: Literal["RAG", "LLM", "WEB"] = Field(description="Node that should handle the user query")
    reasoning: str = Field(description="Reason for selecting this node")

supervisor_model = llm.with_structured_output(
    schema=RouteDecision.model_json_schema(),
    method="json_schema",
)
```

`with_structured_output` guarantees the model's reply matches your schema — no manual JSON parsing,
no risk of the model rambling instead of returning a decision.

### 6.2 Full agent state

```python
class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_reasoning: str
    context: str
    draft_answer: str
    validation_passed: bool
    validation_score: int
    validation_feedback: str
    attempts: int
    used_routes: list[str]
    final_answer: str
```

### 6.3 Supervisor node + router function

The **supervisor node** asks the LLM to pick a route and stores that decision in state. The
**router function** is a *separate*, cheap Python function that just reads `state["route"]` and
returns the node name to go to — this is what `add_conditional_edges` calls.

```python
def supervisor_node(state: AgentState):
    decision = supervisor_model.invoke(supervisor_prompt)   # returns dict-like route/reasoning
    return {
        "route": decision["route"],
        "route_reasoning": decision["reasoning"],
        "attempts": state.get("attempts", 0) + 1,
        "used_routes": state.get("used_routes", []) + [decision["route"]],
        "context": "", "draft_answer": "",     # clear previous attempt
    }

def router(state: AgentState):
    return state["route"]     # "RAG" | "LLM" | "WEB"
```

### 6.4 Validator + retry loop

After any of the three answer-generating nodes runs, a **validator node** scores the answer. A
second routing function decides: pass → finish, fail but under the retry cap → back to supervisor,
fail and out of retries → a "failed" terminal node.

```python
MAX_ATTEMPTS = 3

def validation_router(state: AgentState):
    if state["validation_passed"]:
        return "Final"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "Failed"
    return "Supervisor"          # loop back with feedback
```

### 6.5 Wiring it all together

```python
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("RAG", rag_node)
workflow.add_node("LLM", llm_node)
workflow.add_node("WEB", web_node)
workflow.add_node("Validator", validator_node)
workflow.add_node("Final", final_node)
workflow.add_node("Failed", failed_node)

workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges(
    "Supervisor", router,
    {"RAG": "RAG", "LLM": "LLM", "WEB": "WEB"},
)
workflow.add_edge("RAG", "Validator")
workflow.add_edge("LLM", "Validator")
workflow.add_edge("WEB", "Validator")
workflow.add_conditional_edges(
    "Validator", validation_router,
    {"Final": "Final", "Supervisor": "Supervisor", "Failed": "Failed"},
)
workflow.add_edge("Final", END)
workflow.add_edge("Failed", END)

app = workflow.compile()
```

**Why this pattern matters:** this is the template for almost every serious agentic system — a
node that *decides*, conditional edges that *route*, and a validator that can *loop the graph back*
when quality isn't good enough. Always pass a `recursion_limit` in config when loops are possible,
so a stuck loop fails loudly instead of running forever:

```python
app.invoke(initial_state, {"recursion_limit": 30})
```

---

## 7. Tool Calling — concepts

*(Class 42 — `tool_engineering.ipynb`)*

**Tool calling** = an LLM can decide, on its own, to call an external function to get information or
perform an action it can't do from its own knowledge.

```
User: "What is the weather in Bangalore today?"
      ↓
LLM decides: "I need current weather data."
      ↓
Calls Weather Tool
      ↓
Tool returns: 28°C, cloudy
      ↓
LLM gives final answer
```

Critically: **the LLM never executes the function itself.** It only decides *which* tool to call and
*with what arguments*. Your application code (LangGraph's `ToolNode`, or you manually) actually runs
the function and feeds the result back.

```
LLM = decision maker
Tool = actual executor
```

Generic flow:

```
User → LLM → needs a tool? → yes → Tool Call → Tool Result → LLM → Final Answer
```

Typical things turned into tools: web search, calculator, database query, API calls, a retriever,
sending email, weather/stock APIs, file operations — basically **any capability**.

---

## 8. Node vs Tool — the decision that matters most

*(Class 42 — `tool_engineering.ipynb`)*

This is the single most useful mental model from the tool-engineering class.

| | Node | Tool |
|---|---|---|
| Who decides when it runs | **The graph** (you, the developer) | **The LLM**, at inference time |
| Use for | Deterministic, mandatory steps: validation, guardrails, logging, retries, finalization | Optional capabilities the model may or may not need: calculator, search, DB lookup, weather |

```
Node = workflow step (the graph guarantees when it executes)
Tool = callable capability (the LLM decides whether it's needed)
```

Ask yourself:

- *"Should my workflow guarantee WHEN this runs?"* → **Node**
  (e.g. every answer must pass through a safety validator — you don't want the LLM optionally
  skipping safety checks.)
- *"Should the LLM decide WHETHER it needs this?"* → **Tool**
  (e.g. maybe use a calculator, maybe use web search, maybe answer directly — depends on the
  question.)

**Why not make everything a node?** If you have 50 capabilities and wire them all as nodes with
manual routers, the graph explodes in complexity. Binding them as tools lets the LLM pick the right
one dynamically:

```python
tools = [weather, calculator, retriever, web_search]
model_with_tools = model.bind_tools(tools)
```

**The ideal architecture uses both:**

```
START → Agent Node (LLM + tools) → [Calculator | Web | Retriever tools] → Agent Node
                                                                              ↓
                                                                       Validator Node
                                                                              ↓
                                                                         Final Node → END
```

Agent / Validator / Final are **nodes** (guaranteed steps). Calculator / Web / Retriever are
**tools** (the LLM decides).

---

## 9. Tool Engineering — every way to build a tool

*(Class 42 — `tool_engineering.ipynb`)*

Every tool, no matter how it's built, has the same anatomy:

```
Tool
 ├── Name
 ├── Description
 ├── Input Schema
 └── Actual Functionality
```

The LLM only ever "sees" the name, description, and schema — never the code itself.

### 9.1 `@tool` decorator (the default choice — use this 90% of the time)

```python
from langchain_core.tools import tool

@tool
def add_basic(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

add_basic.name          # "add_basic"
add_basic.description   # "Add two numbers."
add_basic.args          # {'a': {...}, 'b': {...}}
add_basic.invoke({"a": 20, "b": 30})   # 50
```

The **docstring becomes the tool's description** that the LLM reads to decide when to use it — write
it like you're explaining the tool to someone who's never seen your code.

### 9.2 Custom name / description / behavior flags

```python
@tool("calculator")
def add_with_custom_name(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool(
    "multiply_numbers",
    description="Multiply two integers and return the result.",
    return_direct=False,     # if True, tool output is returned straight to the user, skipping another LLM pass
)
def multiply_with_options(a, b):
    return a * b
```

### 9.3 `@tool` with a Pydantic schema (for validated / complex inputs)

```python
from pydantic import BaseModel, Field

class CalculatorInputTest(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")

@tool(args_schema=CalculatorInputTest)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

Passing `"sunny"` for `a` here raises a `ValidationError` — this is exactly how you protect tools
from bad LLM-generated arguments.

### 9.4 `@tool(parse_docstring=True)` — auto-generate arg descriptions from Google-style docstrings

```python
@tool(parse_docstring=True)
def search_with_docstring(query: str, limit: int) -> str:
    """Search documents.

    Args:
        query: Search query entered by the user.
        limit: Maximum number of results.
    """
    return f"Searching for '{query}' with limit={limit}"
```

### 9.5 Async tools — use for any network/API/DB call

```python
@tool
async def get_data_async(url: str) -> str:
    """Fetch data asynchronously."""
    await asyncio.sleep(0.1)
    return f"Data from {url}"

result = await get_data_async.ainvoke({"url": "https://example.com"})
```

### 9.6 `Tool(...)` constructor / `Tool.from_function()` — wrap an existing plain function

```python
from langchain_core.tools import Tool

def simple_search_function(query: str) -> str:
    return f"Searching for {query}"

simple_search_tool = Tool(
    name="simple_search",
    func=simple_search_function,
    description="Search for information.",
)
```

### 9.7 `StructuredTool.from_function()` — explicit structured construction

```python
from langchain_core.tools import StructuredTool

tax_tool = StructuredTool.from_function(
    func=calculate_tax_test,
    name="calculate_tax",
    description="Calculate tax from income and tax rate.",
)
```

### 9.8 Subclassing `BaseTool` — maximum customization

```python
from langchain_core.tools import BaseTool

class MySearchToolTest(BaseTool):
    name: str = "my_search"
    description: str = "Search my custom database."
    args_schema: Type[BaseModel] = SearchInputTest

    def _run(self, query: str) -> str:
        return f"Custom database result for: {query}"
```

### 9.9 `create_retriever_tool()` — turn a RAG retriever into a tool

```python
from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    demo_retriever,
    name="search_company_documents",
    description="Search internal company documents.",
)
```

This is the standard way to plug a vector-store retriever into an **agentic RAG** flow — the LLM
decides *when* to search internal documents, instead of it always happening as a fixed step.

### 9.10 Toolkit — a collection/factory of related tools

```python
from langchain_core.tools import BaseToolkit

class DemoMathToolkit(BaseToolkit):
    def get_tools(self):
        return [toolkit_add, toolkit_multiply]
```

### 9.11 Prebuilt integration tools (e.g. Tavily web search)

```python
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(max_results=3, search_depth="basic", topic="general")
result = tavily_tool.invoke({"query": "What is LangGraph?"})
```

### 9.12 Priority table — what to reach for

| Priority | Method | When |
|---|---|---|
| ⭐⭐⭐⭐⭐ | `@tool` | Default for normal custom functions |
| ⭐⭐⭐⭐⭐ | `@tool` + Pydantic | Complex / validated business input |
| ⭐⭐⭐⭐ | async `@tool` | API / DB / network tools |
| ⭐⭐⭐⭐ | `create_retriever_tool()` | Agentic RAG |
| ⭐⭐⭐⭐ | Prebuilt integrations | Tavily, DB, APIs, etc. |
| ⭐⭐⭐⭐ | MCP | External / shared enterprise tools |
| ⭐⭐⭐ | `ToolRuntime` | Tool needs graph state/context/store |
| ⭐⭐⭐ | Runnable `.as_tool()` | Existing chain/Runnable becomes a tool |
| ⭐⭐ | `StructuredTool.from_function()` | Explicit structured construction |
| ⭐⭐ | `BaseTool` | Advanced custom integration |
| ⭐ | `Tool(...)` / `.from_function()` | Simple explicit wrappers |
| ⭐ | Raw JSON schemas | Provider-level / low-level work |

**Decision tree:** start from "I have a normal Python function" → `@tool`. Simple inputs → plain
`@tool`; complex inputs → `@tool` + Pydantic. Needs network/API/DB → make it `async`. Is it a
retriever → `create_retriever_tool()`. Already have an LCEL Runnable → `.as_tool()`. Needs graph
state/store → `ToolRuntime`. Needs deep customization → `BaseTool`. Lives remotely / shared across
apps → MCP.

### 9.13 How a tool actually gets executed inside a graph: `ToolNode`

Creating a tool and binding it to a model does **not** execute it — `bind_tools()` only lets the
model *generate* a tool call. Something still has to run it. `ToolNode` is that executor:

```python
Create Tool
    ↓
bind_tools()
    ↓
LLM generates tool_calls
    ↓
ToolNode executes the call
    ↓
ToolMessage
    ↓
LLM can continue
```

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

tool_node = ToolNode([subtract])

tool_node_result = tool_node.invoke({
    "messages": [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "subtract",
                "args": {"a": 100, "b": 35},
                "id": "call_subtract_1",
                "type": "tool_call",
            }],
        )
    ]
})
# tool_node_result["messages"][-1].content == "65"
```

`ToolNode` reads the `tool_calls` on the last `AIMessage`, executes each named tool with its
generated args, and appends a `ToolMessage` per call back into the message list.

---

## 10. MCP (Model Context Protocol) tools

*(Class 42 — `tool_engineering.ipynb`)*

MCP lets tools live in a separate server process (possibly written in any language, running anywhere)
and be loaded into LangChain as ordinary `BaseTool` objects.

```
MCP Server
   │
   ├── Tool 1
   ├── Tool 2
   └── Tool 3
        ↓
langchain-mcp-adapters
        ↓
LangChain BaseTools
        ↓
LangGraph Agent
```

**Defining an MCP server** (a separate script, run as its own process):

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add_mcp(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**Loading MCP tools into LangChain, from one or many servers:**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(mcp_server_path.resolve())],
    }
})

mcp_tools = await mcp_client.get_tools()
add_mcp_tool = next(t for t in mcp_tools if t.name == "add_mcp")
await add_mcp_tool.ainvoke({"a": 10, "b": 25})
```

Multiple servers can be registered under different keys in the same client — useful when different
capabilities live in different teams' MCP servers.

**Going the other way** — expose an existing LangChain tool as an MCP tool:

```python
from langchain_mcp_adapters.tools import to_fastmcp

fastmcp_tool = to_fastmcp(langchain_add_for_mcp)
```

---

## 11. The ReAct Agent pattern

*(Class 42 — `ReAct_agent_with_tool_calling.ipynb`)*

**ReAct** ("Reason + Act") is the canonical agent loop: the LLM reasons about what to do, optionally
calls a tool, observes the result, and loops until it has a final answer. LangGraph has prebuilt
helpers (`ToolNode`, `tools_condition`) that make this a ~15-line graph.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

tools = [add, multiply, subtract, divide, power]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
You are a mathematical agent.
...
Do not perform calculations yourself; always use the appropriate tool.
"""

def llm_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("llm_node", llm_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "llm_node")
workflow.add_conditional_edges(
    "llm_node",
    tools_condition,                     # prebuilt: checks if the last AIMessage has tool_calls
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "llm_node")   # after running tools, go back to the LLM

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content=question)]})
```

**What `tools_condition` does internally** (worth understanding, not memorizing):

```python
def tools_condition(state) -> Literal["tools", "__end__"]:
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"
```

It's just "does the last AI message contain tool calls?" — if yes, route to the tools node; if no,
the model produced a final answer, so end.

**The loop shape** is what makes this ReAct: `llm_node → tools → llm_node → tools → ... → llm_node → END`.
Each pass through `tools` executes whatever the model asked for and feeds the result back as a
`ToolMessage`, letting the model decide the *next* step (which might be another tool call, or the
final answer).

Inspecting the full trace is a great debugging habit:

```python
for message in result["messages"]:
    if isinstance(message, AIMessage) and message.tool_calls:
        for call in message.tool_calls:
            print(call["name"], call["args"])
    elif isinstance(message, ToolMessage):
        print("Tool Result:", message.content)
```

---

## 12. Alternative: Planner–Executor pattern (no `bind_tools`)

*(Class 42 — `ReAct_agent_with_tool_calling.ipynb`, second half)*

Instead of letting the LLM emit native tool calls, you can force it to return a structured
**"next action"** object (via `with_structured_output`), and manually dispatch to a **tool registry**
dict. This gives you more control (e.g. guaranteed one-operation-at-a-time execution, easy audit
trail) at the cost of writing more plumbing yourself.

```python
TOOL_REGISTRY = {"add": add, "multiply": multiply, "subtract": subtract, "divide": divide, "power": power}

class NextAction(BaseModel):
    action: Literal["add", "multiply", "subtract", "divide", "power", "finish"]
    x: float | None = None
    y: float | None = None
    reason: str
    final_answer: float | None = None

planner_model = llm.with_structured_output(NextAction)

def planner_node(state: AgentState):
    decision = planner_model.invoke(prompt)   # decides ONE next step, given history so far
    return {"action": decision.action, "x": decision.x, "y": decision.y, "reason": decision.reason}

def planner_router(state: AgentState) -> Literal["execute", "finish"]:
    return "finish" if state["action"] == "finish" else "execute"

def tool_executor_node(state: AgentState):
    selected_tool = TOOL_REGISTRY[state["action"]]
    arguments = {"base": state["x"], "exponent": state["y"]} if state["action"] == "power" \
                else {"a": state["x"], "b": state["y"]}
    result = selected_tool.invoke(arguments)
    history = state.get("history", []).copy()
    history.append({"tool": state["action"], "args": arguments, "result": result})
    return {"last_result": float(result), "history": history, "steps": state.get("steps", 0) + 1}
```

Graph: `START → planner → (execute → tool_executor → planner) loop → finish → final → END`.

**Trust the tool, not the model's arithmetic:** the `final_node` reports `state["last_result"]`
(the actual executed tool output), not anything the LLM claims — this avoids the classic bug where
an LLM "does the math in its head" and gets it slightly wrong even though it called the right tools.

**When to prefer this over ReAct/`bind_tools`:** when you need strict one-step-at-a-time control,
full auditability of every decision, or your provider/model doesn't support native tool calling well.
For most cases, the built-in `bind_tools` + `ToolNode` + `tools_condition` ReAct pattern (Section 11)
is simpler and is the default choice.

---

## 13. Pydantic Fundamentals

*(Class 43 — `pydantic.ipynb`, plus highlights from the companion reference notebook
`complete_pydantic_0_to_204_all_code.ipynb`)*

You've been using `with_structured_output(SomeModel)` and `BaseModel` since Section 6. This section
finally explains *why* that works — Pydantic is the validation engine underneath almost every
structured-output and tool-schema call you've written so far.

### 13.1 What problem does Pydantic solve?

```
Incoming Data
     ↓
Pydantic Model
     ↓
Check: required fields? correct type? valid range? correct format?
     ↓
Valid → continue        Invalid → error
```

**Pydantic is a Python library that lets you define the expected structure of data using normal
Python types, and it automatically validates incoming data against that structure.**

```
Schema defining    = Defining what the data structure should look like.
Schema validation  = Checking whether input data follows the defined structure and rules.
Schema enforcement = Ensuring invalid data is rejected and the rules are actually followed.
```

Without Pydantic, you validate by hand — and this gets unmanageable fast:

```python
def create_user(data):
    if "name" not in data:
        raise ValueError("Missing 'name' field")
    if not isinstance(data["name"], str):
        raise ValueError("Name must be a string")
    if "age" not in data:
        raise ValueError("Age is required")
    if not isinstance(data["age"], int):
        raise ValueError("Age must be an integer")
    if data["age"] < 18:
        raise ValueError("Age must be greater than or equal to 18")
    return {"message": "User created successfully", "user": data}
```

With Pydantic, the same rules are declared once, as the class itself:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(..., ge=18, description="The age of the user, must be 18 or older")

user = User(name="Mayank Chugh", age=49)   # validated on construction
User(name="Ankit Singh", age="forty-nine") # raises ValidationError — "forty-nine" isn't a number
```

**What `BaseModel` gives you for free:** validation, parsing, serialization, JSON Schema generation,
and structured error messages. A plain `class User: ...` (no `BaseModel`) is just an ordinary Python
class with none of this.

### 13.2 Pydantic V1 → V2 method renames

You'll see both spellings in the wild — the class was taught on V2, but older tutorials/code use V1
names. Know the mapping:

| V1 | V2 |
|---|---|
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `.parse_obj()` | `.model_validate()` |
| `.parse_raw()` | `.model_validate_json()` |
| `.schema()` | `.model_json_schema()` |
| `.copy()` | `.model_copy()` |
| `@validator` | `@field_validator` |
| `@root_validator` | `@model_validator` |

### 13.3 `Field()` — validation rules, defaults, and metadata

```python
age: int = Field(ge=18, description="Age of the user")
```

Common constraints: `Field(gt=0)` greater-than, `Field(ge=0)` greater-or-equal, `Field(lt=100)`
less-than, `Field(le=100)` less-or-equal; for strings, `Field(min_length=3, max_length=50)`.

```python
from pydantic import BaseModel, EmailStr, Field, ValidationError

class UserRegistration(BaseModel):
    name: str
    email: EmailStr          # built-in email-format validation
    age: int = Field(ge=18)

try:
    user = UserRegistration(name="Sunny", email="sunnysavita", age=16)
except ValidationError as e:
    print(e)   # tells you exactly which field failed and why
```

```
Pydantic        → detects the problem
ValidationError → represents the problem
try/except      → handles the problem
print(e)        → displays the problem
```

This is dramatically better than dozens of hand-written `if` checks — Pydantic tells you precisely
which field is invalid and why (`email → invalid email`, `age → must be >= 18`), instead of you
guessing.

### 13.4 Required, default, nullable, and optional fields — the four combinations

```python
class User(BaseModel):
    name: str                          # 1. REQUIRED — must be provided
    country: str = "India"             # 2. DEFAULT — falls back to "India" if omitted
    middle_name: str | None            # 3. REQUIRED BUT NULLABLE — must be provided, value can be None
    nickname: str | None = None        # 4. OPTIONAL + NULLABLE — can be omitted AND can be None
```

This distinction trips up beginners constantly — "optional" and "nullable" are two *different* axes
(whether you must supply the key at all, vs. whether `None` is an acceptable value), and Pydantic
lets you control each independently.

### 13.5 Serialization and parsing — the model-object ↔ dict/JSON round trip

```python
user = User(name="Sunny", age=30)          # city defaults to "Bangalore"

user.model_dump()          # -> {'name': 'Sunny', 'age': 30, 'city': 'Bangalore'}  (a dict)
user.model_dump_json()     # -> '{"name":"Sunny","age":30,"city":"Bangalore"}'      (a JSON string)
User.model_json_schema()   # -> the JSON Schema describing this model (this is what LLM structured-
                            #    output / tool-calling providers actually receive)

data = {"name": "Sunny", "age": "30", "city": "Delhi"}
User.model_validate(data)              # parses a dict, coercing "30" -> 30

json_text = '{"name": "Sunny", "age": 30, "city": "Bangalore"}'
User.model_validate_json(json_text)    # parses a JSON string directly

user2 = user.model_copy()                                  # shallow copy
user3 = user.model_copy(update={"city": "Mumbai"})          # copy with one field changed
```

`model_json_schema()` is the bridge to LLMs: when you call `llm.with_structured_output(MyModel)`,
LangChain converts your Pydantic model into exactly this JSON Schema and hands it to the provider's
function-calling / structured-output API — the model in Section 6.1 and every `@tool(args_schema=...)`
in Section 9.3 rely on this same mechanism.

### 13.6 Custom validators: `field_validator` vs `model_validator`

```python
from pydantic import field_validator, model_validator

class Employee(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Name must contain at least 2 characters")
        return value.title()          # validators can also TRANSFORM the value


class Signup(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
```

| `field_validator` | `model_validator` |
|---|---|
| Validates **a single field** | Validates **multiple fields or the whole object** |
| Works on an individual field's value | Works on the complete constructed model |

Rule of thumb: one field's own value → `field_validator`. A rule that compares two-or-more fields
(like password confirmation) → `model_validator`.

### 13.7 Nested schemas

Real-world data is rarely flat. Pydantic models can nest inside each other, and validation recurses
automatically:

```python
class Address(BaseModel):
    city: str
    state: str
    pin_code: int

class Company(BaseModel):
    company_name: str
    department: str

class User(BaseModel):
    name: str
    age: int
    email: EmailStr
    address: Address
    company: Company

User.model_validate({
    "name": "Sunny", "age": 30, "email": "sunny@gmail.com",
    "address": {"city": "Bangalore", "state": "Karnataka", "pin_code": 560001},
    "company": {"company_name": "ABC Technologies", "department": "AI Engineering"},
})
```

```
User
 ├── name, age, email
 ├── Address  → city, state, pin_code
 └── Company  → company_name, department
```

### 13.8 Pydantic × LangGraph / GenAI integration patterns

These are the patterns you'll actually reuse across every agentic project — most of them are things
you already saw in earlier sections, now framed explicitly as "this is Pydantic at work."

**Structured LLM output** (the mechanism behind every `with_structured_output` call since Section 6):

```python
class CityDetails(BaseModel):
    city: str
    country: str
    population: int
    description: str

structured_model = model.with_structured_output(CityDetails)
result = structured_model.invoke("Give details about Bangalore")
result.city
```

**Classification with `Literal`** — constrains the model to a fixed set of labels, exactly like the
`RouteDecision.route` field back in Section 6.1:

```python
from typing import Literal

class Classification(BaseModel):
    category: Literal["billing", "technical", "sales"]
    confidence: float

classifier = model.with_structured_output(Classification)
classifier.invoke("My payment was deducted twice.")
```

**Validated tool input** — the same `args_schema=` pattern from Section 9.3, spelled out again:

```python
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(min_length=3, description="Search query")
    max_results: int = Field(default=5, ge=1, le=10)

@tool(args_schema=SearchInput)
def search_documents(query: str, max_results: int = 5):
    return f"Searching for {query}, max_results={max_results}"
```

**Pydantic as LangGraph state (instead of `TypedDict`)** — LangGraph accepts a Pydantic `BaseModel`
as the state schema too, which buys you real validation (e.g. `attempts` can never go negative) at
the cost of a little extra overhead vs. a plain dict:

```python
class AgentState(BaseModel):
    question: str
    answer: str | None = None
    attempts: int = Field(default=0, ge=0)

def process(state: AgentState):
    return {"answer": "Processed", "attempts": state.attempts + 1}

builder = StateGraph(AgentState)
```

Compare with the `TypedDict` version used everywhere else in this course (Sections 2–12, 15–22):

```python
class AgentState(TypedDict):
    question: str
    answer: str
```

**When to choose which:** `TypedDict` is the default in this course — it's lightweight, has zero
runtime validation overhead per node call, and is what almost every example above uses. Reach for a
Pydantic `BaseModel` as state only when you specifically want field-level validation enforced on
every state update (e.g. a numeric field that must never go negative, or a nested sub-schema).

**A complete agentic slice — routing schema + tool schema + execution, end to end:**

```python
class RouteDecision(BaseModel):
    route: Literal["RAG", "WEB", "CALCULATOR"]
    reason: str

class CalculatorInput(BaseModel):
    a: float
    b: float
    operation: Literal["add", "subtract", "multiply", "divide"]

@tool(args_schema=CalculatorInput)
def calculator(a: float, b: float, operation: str):
    if operation == "add": return a + b
    if operation == "subtract": return a - b
    if operation == "multiply": return a * b
    if operation == "divide":
        if b == 0: raise ValueError("Division by zero")
        return a / b

router_llm = model.with_structured_output(RouteDecision)
decision = router_llm.invoke("User question:\nWhat is 20 multiplied by 10?")

if decision.route == "CALCULATOR":
    result = calculator.invoke({"a": 20, "b": 10, "operation": "multiply"})
```

This is the same routing shape as the Supervisor in Section 6, just compressed to its essence: a
Pydantic model decides *where to go*, a Pydantic-validated tool schema decides *what arguments are
safe to execute with*.

**Bonus — managing API keys/config with `pydantic-settings`**, useful once you have several secrets
(GROQ, OpenAI, Tavily, vector DB, etc.) instead of scattering `os.getenv(...)` calls everywhere:

```python
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    groq_api_key: SecretStr                       # loaded from .env, never printed in plain text
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = Field(default=0, ge=0, le=2)
```

### 13.9 The mental model to keep

```
RAW DATA
    ↓
PYDANTIC SCHEMA
    ↓
TYPE CONVERSION      ("30" -> 30)
    ↓
VALIDATION           (age >= 18? email valid?)
    ↓
PYTHON OBJECT        (or a ValidationError telling you exactly what's wrong)
```

Every `with_structured_output`, every `@tool(args_schema=...)`, every `AgentState(BaseModel)` in
this whole course is this same four-step pipeline, just applied to a different kind of data.

---

## 14. Memory in LangGraph with `Runtime[Context]`

*(Class 43 — `memory_in_langgraph.ipynb`)*

This is a second, slightly different way to wire per-user long-term memory into a graph — worth
knowing *before* the dedicated Memory Management classes (Sections 15–20), because it uses a
different API shape than the `get_store()`/`get_config()` style shown later in Section 19. Both
approaches ultimately read/write the same kind of `Store` — pick whichever your LangGraph version /
project convention favors.

### 14.1 `context_schema` + `Runtime[Context]` instead of `get_config()`/`get_store()`

Instead of calling global getters (`get_store()`, `get_config()`) from inside a node body (Section
19's approach), you can declare a typed **context schema** for the graph and have LangGraph inject a
`Runtime` object directly as a node parameter — it carries both the per-call context (e.g. who the
user is) and the store.

```python
import uuid
from dataclasses import dataclass
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()     # short-term memory (per thread_id)
store = InMemoryStore()            # long-term memory (per user_id)

@dataclass
class Context:
    user_id: str
```

### 14.2 The node: read + write long-term memory via `runtime.store`

```python
def assistant_node(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = ("users", user_id, "memories")
    user_message = state["messages"][-1].content

    # explicit "remember:" trigger phrase — a manual alternative to langmem's
    # automatic extraction (Section 19.1)
    if user_message.lower().startswith("remember:"):
        memory = user_message[len("remember:"):].strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"data": memory})
        print(f"[MEMORY SAVED]: {memory}")

    memories = runtime.store.search(namespace)
    memory_text = "\n".join(memory.value["data"] for memory in memories) or "No saved memories."

    system_prompt = f"""
    You are a helpful assistant.

    Long-term memories about this user:

    {memory_text}
    """

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"],
    ])
    return {"messages": [response]}
```

**Two ways to trigger a memory save, contrasted:**
- Here: the *user* explicitly says `"Remember: ..."` — a simple, predictable, user-controlled trigger.
- `langmem`'s `create_memory_store_manager` (Section 19.1): the *LLM* automatically decides what's
  worth remembering from ordinary conversation, no special phrase needed.

### 14.3 Declaring `context_schema` on the graph, and passing `context=` at call time

```python
builder = StateGraph(MessagesState, context_schema=Context)   # <-- declares the context shape
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

def chat(message, user_id, thread_id):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},   # short-term identity
        context=Context(user_id=user_id),                    # long-term identity — a SEPARATE argument
    )
    return result["messages"][-1].content
```

Note that `user_id` now travels via a dedicated `context=` argument, not buried inside
`config["configurable"]` the way Section 19's version does it.

### 14.4 Verifying the same cross-thread / cross-user behavior as Section 19

```python
chat(message="My current project is Agentic RAG.", user_id="sunny_123", thread_id="thread_1")
chat(message="What is my current project?", user_id="sunny_123", thread_id="thread_1")
chat(message="Remember: my favorite programming language is Python.", user_id="sunny_123", thread_id="thread_1")

# NEW thread, SAME user — long-term memory still recalls it:
chat(message="What is my favorite programming language?", user_id="sunny_123", thread_id="thread_2")
```

```
                 sunny_123
                     ↓
              InMemoryStore
                     ↓
      "favorite language = Python"
                     ↓
             ┌───────┴───────┐
             ↓               ↓
         thread_1         thread_2
```

Memories are namespaced per user, so two different users never see each other's memories:

```
users
│
├── sunny_123
│      └── memories
│           └── favorite language = Python
│
└── rahul_456
       └── memories
            └── favorite language = Java
```

Inspecting the store directly (outside of any graph call):

```python
namespace = ("users", "sunny_123", "memories")
for memory in store.search(namespace):
    print(memory.value["data"])
```

### 14.5 Which style should you use?

| | `get_store()` / `get_config()` (Section 19) | `Runtime[Context]` (this section) |
|---|---|---|
| How a node accesses store/user info | Calls global getters inside the node body | Injected as a typed second parameter |
| Where per-call user identity travels | Inside `config["configurable"]`, alongside `thread_id` | A separate, typed `context=` argument |
| Graph setup | No extra declaration needed | `StateGraph(State, context_schema=Context)` |
| Type safety | None — `config["configurable"]["user_id"]` is just a dict lookup | Strong — `Context` is a dataclass, `runtime.context.user_id` is a real attribute |

Both are valid; `Runtime[Context]` is the more type-safe, explicit style and is worth using in new
projects, while `get_store()`/`get_config()` is what you'll most often see in existing LangGraph
examples and tutorials.

---

## 15. Memory Management — overview

*(Class 44 & 45)*

An LLM is stateless. Without memory, every request is like talking to a stranger:

```
User: My name is Sunny.
AI:   Nice to meet you.
User: What is my name?
```

Without memory, the second question fails — nothing links it to the first turn. With memory:

```
Previous Context + Current Query → Better Response
```

**Two major types:**

```
Memory Management
│
├── Short-Term Memory  (current conversation/session, recent messages)
│      ├── Checkpointer      — saves current thread's state so the same chat can continue
│      ├── Trim              — drop old messages, keep only recent ones (reduce tokens/cost)
│      ├── Delete            — permanently remove selected messages
│      ├── Summarize         — compress old conversation into a short summary
│      └── Custom filtering  — your own rule for what stays in context
│
└── Long-Term Memory   (retained ACROSS conversations/sessions)
       ├── Store.put()       — save a user fact/preference/memory
       ├── Store.get()       — retrieve one exact memory by key/id
       ├── Store.search()    — semantic search over stored memories
       ├── user_id/namespace — separate memories per user or category
       └── Semantic retrieval — retrieve by meaning/similarity, not exact keywords
```

**Why memory management matters** (not just "storing everything"):
- Maintain conversation continuity, handle follow-ups, personalize responses
- Avoid repeated questions, support long-running agent workflows
- **Control context-window usage and reduce token cost** — this is the main engineering constraint

### The problem with "just store everything"

```python
memory = []

def chat(user_input: str) -> str:
    memory.append({"role": "user", "content": user_input})
    response = model.invoke(memory)     # sends the ENTIRE history every time
    memory.append({"role": "assistant", "content": response.content})
    return response.content
```

This works, but as the conversation grows, every request resends the *whole* history: more input
tokens, more cost, more latency, and eventually you can hit the model's context window limit. In
practice, long unmanaged histories also degrade answer quality — very long contexts make some models
less accurate ("lost in the middle" effect), so trimming/summarizing isn't just about cost.

**The fix:** separate *what you store* from *what you send to the LLM*.

---

## 16. Short-term memory: sliding window

*(Class 44 & 45)*

Keep the full history in storage, but only send the most recent N messages to the model.

```python
memory = []
MAX_CONTEXT_MESSAGES = 6

def chat(user_input: str) -> str:
    memory.append({"role": "user", "content": user_input})
    recent_context = memory[-MAX_CONTEXT_MESSAGES:]   # only the tail goes to the LLM
    response = model.invoke(recent_context)
    memory.append({"role": "assistant", "content": response.content})
    return response.content
```

**Trade-off:** simple and cheap, but the model completely loses anything outside the window (e.g.
the user's name mentioned 20 turns ago is gone once it scrolls out).

---

## 17. Short-term memory: summarize + recent messages

*(Class 44 & 45)*

Better strategy: keep recent messages verbatim, but **compress older messages into a running
summary** instead of discarding them outright. Send `summary + recent messages` to the LLM.

```python
memory = []
conversation_summary = ""
full_history = []                      # optional: kept for audit/storage, never sent to LLM directly
MAX_RECENT_MESSAGES = 6

def create_summary(old_summary: str, old_messages: list) -> str:
    conversation_text = format_messages(old_messages)
    prompt = f"""
You maintain conversation memory for an AI assistant.

Existing summary:
{old_summary or "No previous summary."}

Older conversation messages to merge:
{conversation_text}

Create one concise updated summary.
Preserve useful information such as user name, preferences, projects, decisions, important Q&A.
Ignore greetings and unnecessary small talk.
"""
    return model.invoke(prompt).content.strip()

def compress_memory_into_summary():
    global memory, conversation_summary
    if len(memory) <= MAX_RECENT_MESSAGES:
        return
    old_messages = memory[:-MAX_RECENT_MESSAGES]
    memory = memory[-MAX_RECENT_MESSAGES:]
    conversation_summary = create_summary(conversation_summary, old_messages)

def chat(user_input: str) -> str:
    global memory, full_history
    user_message = {"role": "user", "content": user_input}
    memory.append(user_message); full_history.append(user_message)

    messages_for_llm = [
        {"role": "system", "content": f"Previous conversation summary:{conversation_summary or 'No previous summary.'}"},
        *memory,
    ]
    response = model.invoke(messages_for_llm)

    assistant_message = {"role": "assistant", "content": response.content}
    memory.append(assistant_message); full_history.append(assistant_message)

    compress_memory_into_summary()     # only compress after a COMPLETE user/assistant turn
    return response.content
```

**Important detail:** compression only runs after a full user+assistant turn completes — you never
want to split a turn in half between the "recent" window and the "summarized" bucket, or you'd lose
the assistant's reply to a question that's still in the summary.

### How it evolves turn by turn

| Step | `memory` after turn | Covered by summary | `conversation_summary` after turn |
|---|---|---|---|
| M1-M2 | M1 M2 | None | Empty |
| M3-M4 | M1 M2 M3 M4 | None | Empty |
| M5-M6 | M1...M6 | None | Empty |
| M7-M8 | M3...M8 | M1-M2 | "Sunny is the user's name." |
| M9-M10 | M5...M10 | M1-M4 | "...name and lives in Bengaluru." |
| M11-M12 | M7...M12 | M1-M6 | "...and is learning LangGraph." |

The window keeps the last 6 messages verbatim; everything that scrolls out gets folded into the
summary instead of vanishing.

This logic is cleanly wrapped as a reusable class:

```python
class ConversationMemory:
    def __init__(self, model, max_recent_messages=6):
        self.model = model
        self.max_recent_messages = max_recent_messages
        self.memory = []
        self.summary = ""
        self.full_history = []

    def compress(self):
        if len(self.memory) <= self.max_recent_messages:
            return
        old_messages = self.memory[:-self.max_recent_messages]
        self.memory = self.memory[-self.max_recent_messages:]
        self.summary = self.create_summary(old_messages)
    # ...add_user_message / add_assistant_message / get_recent_conversation similarly
```

---

## 18. Long-term memory: vector-store RAG + conversation memory

*(Class 45)*

Combining RAG (facts from documents) with conversation memory (facts from the chat itself) lets an
assistant answer both "what does the policy say?" and "what did I tell you earlier?"

### 16.1 Minimal version with FAISS

```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(embedder.get_embedding_dimension())
documents = []

def add_document(text):
    index.add(np.array([embed(text)]))
    documents.append(text)

def retrieve(query, k=2):
    distances, ids = index.search(np.array([embed(query)]), k)
    return [documents[i] for i in ids[0] if i != -1]

def generate_answer(query, retrieved_documents, memory):
    memory_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in memory) or "No previous conversation."
    documents_text = "\n".join(retrieved_documents) or "No relevant documents found."
    prompt = f"""
You are a helpful AI assistant.

Previous conversation:
{memory_text}

Relevant documents:
{documents_text}

Current user question:
{query}

Answer using the relevant documents and previous conversation when useful.
If the answer is not available in the provided documents, say you do not know.
"""
    return model.invoke(prompt).content.strip()
```

Each turn: **retrieve** relevant docs → **generate** an answer using docs + memory → **append**
`(query, answer)` to `memory` for the next turn.

### 16.2 Production-shaped version: `RAGAssistant` with query rewriting

A more robust RAG+memory assistant adds a **query rewriting** step — turning a conversational,
context-dependent question ("what about the other one?") into a standalone search query before
hitting the retriever. This matters because raw follow-up questions often retrieve poorly on their
own.

```python
class RAGAssistant:
    def __init__(self, model, retriever, max_recent_messages=6):
        self.model = model
        self.retriever = retriever
        self.memory = ConversationMemory(model=model, max_recent_messages=max_recent_messages)

    def rewrite_query(self, user_query):
        prompt = f"""
You rewrite conversational questions into standalone search queries.

Conversation summary:
{self.memory.summary or "No previous summary."}

Recent conversation:
{self.memory.get_recent_conversation()}

Current user question:
{user_query}

Rewrite the current question as a complete, standalone search query.
Return only the rewritten query.
"""
        return self.model.invoke(prompt).content.strip()

    def chat(self, user_input):
        self.memory.add_user_message(user_input)
        standalone_query = self.rewrite_query(user_input)
        documents = self.retriever.invoke(standalone_query)
        answer = self.generate_answer(standalone_query, documents)
        self.memory.add_assistant_message(answer)
        self.memory.compress()
        return {"answer": answer, "standalone_query": standalone_query, "documents": documents}
```

Vector store setup (LangChain `InMemoryVectorStore` + OpenAI embeddings):

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

rag = RAGAssistant(model=model, retriever=retriever, max_recent_messages=6)
```

---

## 19. Production memory in LangGraph: Checkpointer + Store + langmem

*(Class 45)*

The hand-rolled memory classes above are great for learning, but LangGraph has **first-class
building blocks** for exactly this:

- **Checkpointer** → short-term memory. Persists a *thread's* state after every node runs, keyed by
  `thread_id`. This is what lets you call `graph.invoke(...)` again later on the same thread and
  have it "remember" everything.
- **Store** → long-term memory. A key-value + semantic-search store, keyed by `namespace` (typically
  including `user_id`), so memories persist *across* threads/sessions for the same user.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()                 # short-term (per-thread)
store = InMemoryStore(index={                  # long-term (per-user, semantic search enabled)
    "dims": 1536,
    "embed": "openai:text-embedding-3-small",
})
```

### 17.1 Automatic long-term memory extraction with `langmem`

`langmem`'s `create_memory_store_manager` watches conversation messages and automatically decides
what's worth remembering long-term (skipping greetings/small talk), writing straight into the
`Store`.

```python
from langmem import create_memory_store_manager

memory_manager = create_memory_store_manager(
    "openai:gpt-4.1-mini",
    namespace=("memories", "{user_id}"),     # one namespace per user
    instructions="""
Extract useful long-term information about the user.
Store: user preferences, important personal facts, goals, projects, decisions.
Ignore greetings, temporary questions, and unnecessary small talk.
""",
)
```

### 17.2 Wiring both into a graph

```python
from langgraph.config import get_store, get_config

def chatbot(state: MessagesState):
    config = get_config()
    current_store = get_store()
    user_id = config["configurable"]["user_id"]

    user_query = state["messages"][-1].content
    memories = current_store.search(("memories", user_id), query=user_query, limit=5)
    memory_text = "\n".join(str(m.value) for m in memories) or "No relevant long-term memories."

    system_message = {"role": "system", "content": f"""
You are a helpful AI assistant.

Relevant long-term memories about the user:
{memory_text}

Use these memories only when relevant.
"""}

    response = model.invoke([system_message, *state["messages"]])
    return {"messages": [response]}

def save_memory(state: MessagesState, config):
    memory_manager.invoke({"messages": state["messages"]}, config=config)
    return {}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("save_memory", save_memory)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "save_memory")
builder.add_edge("save_memory", END)

graph = builder.compile(checkpointer=checkpointer, store=store)
```

Calling the graph requires **both** a `thread_id` (short-term/checkpointer) **and** a `user_id`
(long-term/store) in config:

```python
def chat(user_id, thread_id, message):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return result["messages"][-1].content
```

**Key proof-of-concept result from the class:** if you `chat(...)` on `thread_1`, then start a
**brand-new** `thread_2` for the *same* `user_id` and ask "what do you remember about me?", the
model still answers correctly — because although the checkpointer's thread history reset, the
**Store's long-term memory is keyed by `user_id`, not `thread_id`**, so it persists across threads.

```
Short-term (checkpointer) → scoped to thread_id → resets per conversation
Long-term  (store)        → scoped to user_id   → persists across ALL conversations
```

Inspecting each kind of memory directly:

```python
# Short-term: what's in one thread's history?
snapshot = graph.get_state({"configurable": {"user_id": user_id, "thread_id": thread_id}})
messages = snapshot.values.get("messages", [])

# Long-term: what has been learned about this user?
memories = store.search(("memories", user_id), limit=100)
```

---

## 20. Persistent memory with SQLite

*(Class 45)*

`InMemorySaver`/`InMemoryStore` vanish when the process restarts — fine for demos, not for
production. Swap in a real database.

### 18.1 SQLite checkpointer (short-term, survives restarts)

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

checkpoint_connection = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_connection)
```

Everything else about using the graph stays the same — only the checkpointer backing storage changed.

### 18.2 Hand-rolled SQLite long-term memory store

When you want full control over long-term storage (dedup, custom schema, etc.) instead of using
`langmem`, you can write a tiny SQLite-backed store yourself:

```python
class SQLiteLongTermMemory:
    def __init__(self, db_path="long_term_memory.db"):
        self.db_path = db_path
        self.create_table()

    def create_table(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_memory(self, user_id, content):
        if content in self.get_memories(user_id):   # avoid duplicate memories
            return
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO memories (memory_id, user_id, content) VALUES (?, ?, ?)",
                (str(uuid.uuid4()), user_id, content),
            )

    def get_memories(self, user_id):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT content FROM memories WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [row[0] for row in rows]
```

### 18.3 Manual extraction with structured output (an alternative to `langmem`)

```python
class ExtractedMemories(BaseModel):
    memories: List[str] = Field(default_factory=list, description="Important long-term info learned about the user.")

memory_extractor = model.with_structured_output(ExtractedMemories)

def extract_memories(user_message):
    prompt = f"""
Analyze the following user message.

User message:
{user_message}

Extract only useful long-term information about the user (preferences, facts, goals, projects,
skills, decisions). Do NOT store greetings, temporary questions, or general knowledge questions.
If there is nothing useful to remember, return an empty list.
"""
    return memory_extractor.invoke(prompt).memories
```

### 18.4 Graph wiring — same pattern, different backing stores

```python
def chatbot(state: MessagesState, config):
    user_id = config["configurable"]["user_id"]
    memories = long_term_memory.get_memories(user_id)
    memory_text = "\n".join(f"- {m}" for m in memories) or "No saved long-term memories."
    system_message = SystemMessage(content=f"...{memory_text}...")
    response = model.invoke([system_message, *state["messages"]])
    return {"messages": [response]}

def save_long_term_memory(state: MessagesState, config):
    user_id = config["configurable"]["user_id"]
    latest_user_message = next((m.content for m in reversed(state["messages"]) if m.type == "human"), None)
    if not latest_user_message:
        return {}
    for memory in extract_memories(latest_user_message):
        long_term_memory.add_memory(user_id, memory)
    return {}

graph = builder.compile(checkpointer=checkpointer)   # store not needed — using our own SQLite class
```

**Verified behavior from the class's tests:**
- Same `user_id`, same `thread_id` → checkpointer remembers the conversation turn-by-turn.
- Same `user_id`, **different** `thread_id` → conversation history is fresh, but SQLite long-term
  memory (keyed by `user_id`) still recalls facts like favorite language / current project.
- **Different** `user_id` → no memories are shared; a new user starts with a clean slate.

---

## 21. Human-in-the-Loop (HITL)

*(Class 46)*

Some actions are too risky to let an LLM execute autonomously (refunds, deletions, sending money,
etc.). The pattern: **the LLM is allowed to propose a decision, but not execute it.** The workflow
pauses for human approval, and only after approval does the application actually perform the action.

```
User Request
     ↓
    LLM
     ↓
Structured Proposed Action
{ action: "refund", customer: "CUST-101", amount: 5000, reason: "..." }
     ↓
HUMAN REVIEW
     ↓
Approve?  yes/no
   /        \
 yes          no
  ↓            ↓
Execute      Stop
```

### 19.1 Part 1 — the simple/manual version (no LangGraph state pausing)

This version blocks on `input()` in plain Python — useful to understand the *shape* of HITL before
adding LangGraph's real pause/resume mechanism.

```python
class ProposedAction(BaseModel):
    action: Literal["refund", "reject", "manual_review"]
    customer: str = Field(description="Customer ID")
    amount: float = Field(description="Refund amount")
    reason: str = Field(description="Reason for the proposed action")

structured_model = model.with_structured_output(ProposedAction)

def ai_assistant(user_request):
    prompt = f"""
You are a customer support AI agent.
Analyze the following customer request: {user_request}
Decide the appropriate action: refund / reject / manual_review.
Do not execute the action. Only propose the action for human approval.
"""
    return structured_model.invoke(prompt)

def human_review(action):
    print("Action:", action.action, "| Customer:", action.customer, "| Amount:", action.amount)
    decision = input("\nApprove this action? (yes/no): ").strip().lower()
    return decision == "yes"

def execute_action(action):
    # the ONLY function that performs a real side effect — only ever called after approval
    if action.action == "refund":
        print(f"Refund of ₹{action.amount} processed for {action.customer}.")
    elif action.action == "reject":
        print(f"Request rejected for {action.customer}.")
    else:
        print(f"Request for {action.customer} sent for manual review.")

def run():
    proposed_action = ai_assistant(input("User Request: "))
    if human_review(proposed_action):
        execute_action(proposed_action)
    else:
        print("Action rejected by human.")
```

**Limitation:** this only "pauses" because Python's `input()` blocks the thread. It can't be paused,
persisted, and resumed later by a different process (e.g. a web request approved hours later by a
manager) — that's what Part 2 solves.

### 19.2 Part 2 — real pause/resume with `interrupt()` + `Command(resume=...)`

LangGraph can **durably suspend a graph mid-execution**, hand a payload back to the caller, and
later **resume exactly where it left off** — even from a different process — as long as a
**checkpointer** is attached.

```
User Request → LLM Analyze → Proposed Action → interrupt() → GRAPH PAUSED
                                                                    ↓
                                                             Human Approval
                                                              /          \
                                                            Yes            No
                                                             ↓              ↓
                                              Command(resume=True)  Command(resume=False)
                                                             ↓              ↓
                                                          Execute        Reject
```

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class AgentState(TypedDict):
    user_request: str
    action: str
    customer: str
    amount: float
    reason: str
    approved: bool          # set once the graph is resumed
    final_message: str

def analyze_request(state: AgentState):
    result = structured_model.invoke(prompt)   # LLM proposes, does NOT execute
    return {"action": result.action, "customer": result.customer,
            "amount": result.amount, "reason": result.reason}

def human_approval(state: AgentState):
    # interrupt(payload) suspends execution HERE. The payload is returned to the
    # caller via result["__interrupt__"]. On the NEXT graph.invoke(Command(resume=X),
    # config=same_thread_id), execution resumes from this exact line, and
    # interrupt() returns X as if it were a normal function call.
    decision = interrupt({
        "message": "Please review the AI proposed action.",
        "action": state["action"], "customer": state["customer"],
        "amount": state["amount"], "reason": state["reason"],
    })
    return {"approved": decision}

def approval_router(state: AgentState):
    return "execute" if state["approved"] else "reject"

def execute_action(state: AgentState):
    ...
    return {"final_message": message}

def reject_action(state: AgentState):
    return {"final_message": "Action rejected by human reviewer."}
```

Graph wiring:

```python
builder = StateGraph(AgentState)
builder.add_node("analyze", analyze_request)
builder.add_node("human_approval", human_approval)
builder.add_node("execute", execute_action)
builder.add_node("reject", reject_action)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "human_approval")
builder.add_conditional_edges(
    "human_approval", approval_router,
    {"execute": "execute", "reject": "reject"},
)
builder.add_edge("execute", END)
builder.add_edge("reject", END)

checkpointer = InMemorySaver()      # REQUIRED — without it there's nothing to resume from
graph = builder.compile(checkpointer=checkpointer)
```

**Running it — two separate `invoke()` calls, same `thread_id`:**

```python
config = {"configurable": {"thread_id": "refund_thread_1"}}

# 1st call: runs "analyze", hits interrupt() inside "human_approval", and STOPS there.
result = graph.invoke({...initial state...}, config=config)
print(result["__interrupt__"])      # the payload passed to interrupt() — show this to a human

# ...later, possibly in a totally different process/request...
approved = input("Approve action? (yes/no): ").strip().lower() == "yes"

# 2nd call: resumes exactly at the interrupt() call, which now returns `approved`.
final_result = graph.invoke(Command(resume=approved), config=config)
print(final_result["final_message"])
```

**Why this matters over the Part 1 version:**
- The pause is **durable** — it's backed by the checkpointer, not a blocked Python thread. You can
  shut down the process and resume hours/days later as long as you keep the `thread_id`.
- The approval can come from **anywhere** — a web form, a Slack button, a queue — `interrupt`/
  `Command(resume=...)` doesn't care about the source, only that the same `thread_id` is used to
  resume.
- `result["__interrupt__"]` is the hook a UI uses to render "here's what the AI wants to do, please
  approve/reject."

---

## 22. Subgraphs

*(Class 47)*

A **subgraph** is just a compiled LangGraph graph that gets used *inside* another graph — either
as a node directly, or invoked manually from within a node. Subgraphs let you build reusable,
independently-testable units of workflow (e.g. "the research pipeline") and compose them.

```
Example 1 (shared state)
START → validate_request → [check_inventory → calculate_price] → final_response → END

Example 2 (Send-based fan-out, different state per level)
                              ┌→ worker(topic 1) → research subgraph ─┐
START → prepare_topics → Send ├→ worker(topic 2) → research subgraph ─┼→ combine_results → END
                              └→ worker(topic 3) → research subgraph ─┘
```

### 20.1 Pattern A — shared state: add a compiled subgraph directly as a node

When the parent graph and the subgraph use the **exact same state schema**, you can compile the
subgraph and register it as a node in the parent with `add_node` — state passes through
automatically, no manual translation needed.

```python
class OrderState(TypedDict):
    product: str
    quantity: int
    inventory_status: str
    total_price: float
    final_response: str

# --- build the subgraph ---
def check_inventory(state: OrderState):
    product = PRODUCT_CATALOG[state["product"]]
    is_available = product["stock"] >= state["quantity"]
    return {"inventory_status": "Available" if is_available else "Unavailable"}

def calculate_price(state: OrderState):
    if state["inventory_status"] != "Available":
        return {"total_price": 0.0}
    unit_price = PRODUCT_CATALOG[state["product"]]["unit_price"]
    return {"total_price": state["quantity"] * unit_price}

order_subgraph_builder = StateGraph(OrderState)
order_subgraph_builder.add_node("check_inventory", check_inventory)
order_subgraph_builder.add_node("calculate_price", calculate_price)
order_subgraph_builder.add_edge(START, "check_inventory")
order_subgraph_builder.add_edge("check_inventory", "calculate_price")
order_subgraph_builder.add_edge("calculate_price", END)
order_subgraph = order_subgraph_builder.compile()          # <-- a compiled graph, usable as a node

# --- build the parent graph, nesting the subgraph ---
order_builder = StateGraph(OrderState)
order_builder.add_node("validate_request", validate_request)
order_builder.add_node("order_processing", order_subgraph)  # <-- pass the COMPILED subgraph itself
order_builder.add_node("final_response", create_final_response)

order_builder.add_edge(START, "validate_request")
order_builder.add_edge("validate_request", "order_processing")
order_builder.add_edge("order_processing", "final_response")
order_builder.add_edge("final_response", END)

order_graph = order_builder.compile()
```

`create_final_response` can read `state["inventory_status"]` and `state["total_price"]` — fields
that were written **inside** the subgraph — proving state flows transparently back into the parent
because both share `OrderState`.

**Visualizing nested subgraphs:** use `xray=True` to expand the subgraph's internal nodes in the
diagram; without it, the subgraph shows as a single opaque node.

```python
print(order_graph.get_graph(xray=True).draw_mermaid())
```

### 20.2 Pattern B — different state schemas + parallel fan-out with `Send`

When the parent and subgraph state shapes **differ**, or you need to run the *same* subgraph many
times in parallel (once per item in a list), you don't register the subgraph as a static node.
Instead, a **worker** node manually `.invoke()`s the subgraph, acting as an adapter between schemas.

```python
class ParentState(TypedDict):
    topics: list[str]
    # operator.add reducer: when multiple parallel branches each return a
    # "research_results" update, LangGraph CONCATENATES the lists instead of one
    # overwriting another. Required for Send-based fan-out.
    research_results: Annotated[list[str], operator.add]
    final_answer: str

class ResearchWorkerInput(TypedDict):
    topic: str                     # the payload Send(...) delivers to each worker call

class ResearchState(TypedDict):    # state used ONLY inside the subgraph
    topic: str
    research: str
    summary: str
```

Subgraph (compiled, but **never added as a node** — invoked manually instead):

```python
research_builder = StateGraph(ResearchState)
research_builder.add_node("research_topic", research_topic)
research_builder.add_node("summarize_topic", summarize_topic)
research_builder.add_edge(START, "research_topic")
research_builder.add_edge("research_topic", "summarize_topic")
research_builder.add_edge("summarize_topic", END)
research_subgraph = research_builder.compile()
```

The **worker** node is the adapter — it receives just `{"topic": ...}` (from `Send`) and manually
invokes the subgraph with a full `ResearchState` payload:

```python
def research_worker(state: ResearchWorkerInput):
    topic = state["topic"]
    subgraph_result = research_subgraph.invoke({"topic": topic, "research": "", "summary": ""})
    # returning a single-item list here gets CONCATENATED with other parallel
    # workers' lists, thanks to the operator.add reducer on research_results
    return {"research_results": [f"## {topic}\n\n{subgraph_result['summary']}"]}
```

`Send` is how one list turns into N parallel node executions at runtime — a conditional-edge
function returns a **list of `Send` objects** instead of a single next-node name:

```python
def distribute_topics(state: ParentState):
    return [Send("research_worker", {"topic": topic}) for topic in state["topics"]]

research_parent_builder.add_conditional_edges(
    "prepare_topics",
    distribute_topics,
    ["research_worker"],           # tells LangGraph which node names this edge may route to
)
```

Once every parallel `research_worker` branch finishes, `combine_results` sees the fully merged list:

```python
def combine_results(state: ParentState):
    combined_research = "\n\n".join(state["research_results"])
    response = model.invoke(f"Combine these research summaries...\n\n{combined_research}")
    return {"final_answer": response.content}
```

Full parent wiring:

```python
research_parent_builder = StateGraph(ParentState)
research_parent_builder.add_node("prepare_topics", prepare_topics)
research_parent_builder.add_node("research_worker", research_worker)
research_parent_builder.add_node("combine_results", combine_results)

research_parent_builder.add_edge(START, "prepare_topics")
research_parent_builder.add_conditional_edges("prepare_topics", distribute_topics, ["research_worker"])
research_parent_builder.add_edge("research_worker", "combine_results")
research_parent_builder.add_edge("combine_results", END)

research_graph = research_parent_builder.compile()

# seed research_results as [] — reducers COMBINE updates, they don't invent an initial value
result = research_graph.invoke({"topics": [...], "research_results": []})
```

### 20.3 Pattern A vs Pattern B — when to use which

| | Pattern A (shared state, node) | Pattern B (different state, `Send`) |
|---|---|---|
| State schema | Parent and subgraph share one schema | Parent, worker, and subgraph each have their own schema |
| How it's wired | `add_node("name", compiled_subgraph)` | Manually `.invoke()`d inside a worker node |
| Diagram | `xray=True` expands the subgraph's internal nodes automatically | Parent diagram does NOT show the subgraph's nodes (LangGraph has no static knowledge of a manual `.invoke()` call) |
| Runs how many times | Once | Once per item, in parallel, via `Send` |
| Use for | Grouping related sequential steps into a reusable unit | Reusing one subgraph across a dynamically-sized list of independent tasks (map-reduce style) |

**Common mistakes to watch for:**
- Forgetting the `operator.add` reducer on a field that multiple parallel `Send` branches write to
  → later branch silently overwrites earlier ones instead of combining.
- Forgetting to list allowed destination node names in `add_conditional_edges(..., ["research_worker"])`.
- Expecting a manually-`.invoke()`d subgraph (Pattern B) to show up automatically in `xray=True` —
  it won't, because LangGraph only knows about subgraphs registered via `add_node`.

---

## 23. Multi-Agent Patterns

*(Class 49–50 — `multiagentic-flow.ipynb`)*

Everything up to Section 22 builds single graphs where one designer wires every node/edge by hand.
This class collects **eight independent multi-agent patterns** — different ways to let multiple LLM
"agents" or specialized LLM nodes cooperate, each with a different control style. The patterns are
independent (each builds and compiles its own graph), so you can study/run them one at a time.

### 23.0 A crucial distinction: Role ≠ Agent

A node named `technical_agent` is not automatically "agentic." Something only deserves to be called
an agent when it **observes state, dynamically chooses an action (e.g. which tool to call), and
decides when to stop** — a real decision loop, not just a differently-prompted LLM call.

```
Fixed shape (a workflow):        Agentic shape (a real agent):
LLM Router                        LLM
  ↓                                 ↓
Fixed specialist node             observes state, picks an action
  ↓                                 ↓
LLM response                     ├── call tool A
                                  ├── call tool B
                                  └── finish
                                     ↺ (loops until it decides to stop)
```

Several patterns below (Router, plain Supervisor/Handoff workers) are really *LLM workflow nodes*
with agent-sounding names — they become true agents once given their own tools via `create_agent`
(as in Subagents and Swarm).

### 23.1 Pattern 1 — Router

**Control style:** a router LLM call picks exactly one specialist; only that specialist runs.

```text
User → Router → One Specialist → END
```

```python
class RouterDecision(BaseModel):
    route: Literal["technical", "billing", "general"]

router_model = model.with_structured_output(RouterDecision)

class RouterState(TypedDict):
    query: str
    route: str
    response: str

def router_node(state: RouterState):
    decision = router_model.invoke(f"Route this query...\n{state['query']}")
    return {"route": decision.route}

def route_to_specialist(state: RouterState):
    return state["route"]          # the conditional-edge function just replays the decision

builder = StateGraph(RouterState)
builder.add_node("router", router_node)
builder.add_node("technical", technical_agent)
builder.add_node("billing", billing_agent)
builder.add_node("general", general_agent)
builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_to_specialist,
    {"technical": "technical", "billing": "billing", "general": "general"})
builder.add_edge("technical", END)
builder.add_edge("billing", END)
builder.add_edge("general", END)
```

2 model calls total: the routing decision, then one specialist reply. This is the same
"structured decision + `add_conditional_edges`" shape as Section 6's supervisor/router — routing is
the single most common LLM-driven control-flow primitive in this whole course.

### 23.2 Pattern 2 — Subagents / Agent-as-Tool

**Control style:** a main coordinator agent (built with `create_agent`, LangChain's tool-calling
agent helper) keeps control the whole time and calls other agents *as tools*.

```text
User → Main Agent → Specialist Tools → Main Agent → Final Answer
```

```python
from langchain.agents import create_agent

research_subagent = create_agent(model=model, tools=[], system_prompt="You are a research specialist...")
math_subagent = create_agent(model=model, tools=[], system_prompt="You are a math specialist...")

def get_last_message_content(result):
    return result["messages"][-1].content

@tool
def ask_research_agent(task: str) -> str:
    """Delegate a research task to the research specialist."""
    result = research_subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return get_last_message_content(result)

main_delegation_agent = create_agent(
    model=model,
    tools=[ask_research_agent, ask_math_agent, ask_writing_agent],
    system_prompt="You are the main coordinator. Delegate work to the appropriate specialist tools...",
    name="main_coordinator",
)
```

The **coordinator** is clearly agentic (its internal ReAct loop decides which/how many subagent
tools to call, and in what order). The subagents themselves have *no tools*, so — per Section 23.0 —
they're really specialized LLM workers wrapped in agent objects, not fully autonomous agents yet.
Model-call count is variable, driven entirely by the coordinator's own tool-calling decisions.

### 23.3 Pattern 3 — Supervisor

**Control style:** one central supervisor node repeatedly re-decides the next worker after seeing
accumulated results, looping until it chooses `FINISH`.

```text
Supervisor → Worker → Supervisor → Another Worker → ... → Finish
```

```python
class SupervisorDecision(BaseModel):
    next_agent: Literal["research_agent", "math_agent", "writer_agent", "FINISH"]
    instruction: str

class SupervisorState(TypedDict):
    task: str
    next_agent: str
    instruction: str
    worker_outputs: Annotated[list[str], operator.add]   # reducer: workers APPEND, never overwrite
    final_answer: str
    supervisor_turns: int

def supervisor(state: SupervisorState):
    supervisor_turns = state.get("supervisor_turns", 0) + 1
    completed_work = "\n\n".join(state.get("worker_outputs", []))

    if supervisor_turns >= 6 and completed_work:      # safety valve against infinite delegation
        return {"next_agent": "FINISH", "instruction": "Maximum turns reached...", "supervisor_turns": supervisor_turns}

    decision = supervisor_router.invoke(f"""
Choose the next worker given the task and work completed so far...
{completed_work or "No worker has completed any work yet."}
""")
    return {"next_agent": decision.next_agent, "instruction": decision.instruction, "supervisor_turns": supervisor_turns}
```

Graph shape: `START → supervisor → (conditional) {research_agent|math_agent|writer_agent|finish}`,
with every worker edging **back to `supervisor`** — a loop, not a straight line. Two independent
safety nets are stacked: a manual `supervisor_turns` counter *and* LangGraph's own `recursion_limit`
passed at invoke time (`graph.invoke(state, config={"recursion_limit": 20})`).

**Supervisor is the "central controller" archetype:** the supervisor is agentic (observe → decide →
route → observe again); the workers are specialized LLM calls, not autonomous agents, unless they're
also given their own tools.

### 23.4 Pattern 4 — Handoff

**Control style:** a triage node makes one decision and **transfers control outright** using
`Command(goto=...)`, instead of going through a conditional-edge lookup table.

```text
User → Triage ─┬→ Billing → END
               └→ Technical → END
```

```python
class HandoffState(MessagesState):
    active_agent: str

def triage_handoff_agent(state: HandoffState) -> Command[Literal["billing_agent", "technical_agent"]]:
    latest_user_message = state["messages"][-1].content
    decision = handoff_router_model.invoke(f"Choose which specialist should handle...\n{latest_user_message}")

    return Command(
        goto=decision.target,                          # <-- picks the next node directly
        update={
            "active_agent": decision.target,
            "messages": [AIMessage(content="Handoff reason: " + decision.reason)],
        },
    )
```

`Command` fuses "decide" and "route" into a single return value — the node both updates state
*and* tells the graph engine exactly which node runs next, so no separate `add_conditional_edges`
routing table is needed for the triage step. Compare with Router (Pattern 1): Router uses a
conditional-edge lookup table computed *after* the decision node runs; Handoff embeds the routing
decision directly inside the node's own return value.

### 23.5 Pattern 5 — Swarm

**Control style:** fully decentralized — every agent carries its own "handoff tool" and decides
*for itself*, at runtime, whether to transfer control to the other agent. No permanent supervisor.

```text
Math Agent ⇄ Writer Agent
```

```python
from langgraph_swarm import create_handoff_tool, create_swarm

math_agent = create_agent(
    model=model,
    tools=[create_handoff_tool(agent_name="writer_agent", description="Transfer when polished writing is needed.")],
    system_prompt="You are a math specialist... Transfer to writer_agent when...",
    name="math_agent",
)
writer_agent = create_agent(
    model=model,
    tools=[create_handoff_tool(agent_name="math_agent", description="Transfer when calculations are needed.")],
    system_prompt="You are a professional writer... Transfer to math_agent when...",
    name="writer_agent",
)

workflow = create_swarm([math_agent, writer_agent], default_active_agent="math_agent")
swarm_graph = workflow.compile(checkpointer=InMemorySaver())   # checkpointer required: tracks "active agent" per thread

result = swarm_graph.invoke(
    {"messages": [{"role": "user", "content": "Calculate 125 * 32 and present it professionally."}]},
    config={"configurable": {"thread_id": "swarm_thread_1"}},
)
```

**Supervisor vs. Swarm, the key contrast:** Supervisor = one central LLM decides every step for
everyone; Swarm = agents directly hand off to each other, with control genuinely distributed. Both
agents here have real tools (their handoff tools) and genuinely decide whether to use them — this is
the strongest "true multi-agent" example in the notebook, per the Section 23.0 test.

### 23.6 Pattern 6 — Hierarchical Supervisor

**Control style:** a two-level "graph of graphs." A top supervisor picks one **team subgraph**; that
team runs its own fixed internal pipeline and returns a combined report.

```text
Top Supervisor ─┬→ Research Team Subgraph → END
                └→ Business Team Subgraph → END
```

```python
class TeamState(TypedDict):
    task: str
    team_output: str

def research_team_node(state: TeamState):
    research = model.invoke(f"Research/explain: {state['task']}")
    writer = model.invoke(f"Convert this research into a concise report:\n{research.content}")
    return {"team_output": writer.content}

research_team_builder = StateGraph(TeamState)
research_team_builder.add_node("research_team", research_team_node)
research_team_builder.add_edge(START, "research_team")
research_team_builder.add_edge("research_team", END)
research_team_graph = research_team_builder.compile()      # a complete, independently-runnable graph

class HierarchicalState(TypedDict):
    task: str
    selected_team: str
    final_answer: str

def call_research_team(state: HierarchicalState):
    result = research_team_graph.invoke({"task": state["task"], "team_output": ""})   # sub-invocation
    return {"final_answer": result["team_output"]}

builder = StateGraph(HierarchicalState)
builder.add_node("top_supervisor", top_supervisor)
builder.add_node("research_team", call_research_team)
builder.add_node("business_team", call_business_team)
builder.add_edge(START, "top_supervisor")
builder.add_conditional_edges("top_supervisor", hierarchical_route,
    {"research_team": "research_team", "business_team": "business_team"})
builder.add_edge("research_team", END)
builder.add_edge("business_team", END)
```

This is a lighter-weight cousin of Section 20's subgraph **Pattern A** (shared-state, `add_node`
composition) — except here the compiled team graph is invoked *manually inside a wrapper node*
(`call_research_team`) rather than registered directly as a node, because the team's own `TeamState`
schema differs from the parent's `HierarchicalState`. Expected model calls: 3 (one team-selection
call, plus two calls inside whichever team gets picked).

### 23.7 Pattern 7 — Parallel Collaboration with `Send`

**Control style:** a coordinator LLM plans a *variable-length* list of subtasks; `Send` spawns one
parallel worker execution per subtask; a reducer merges all outputs; an aggregator combines them.

```text
Coordinator → Dynamic Workers (N, in parallel) → Reducer merge → Aggregator → END
```

```python
class AgentTask(BaseModel):
    role: Literal["research", "security", "business"]
    task: str

class AgentPlan(BaseModel):
    tasks: list[AgentTask]                 # the LLM decides how MANY tasks, and which roles

class CollaborationState(TypedDict):
    user_task: str
    tasks: list[AgentTask]
    outputs: Annotated[list[str], operator.add]      # merges N parallel workers' outputs
    final_answer: str

def collaboration_coordinator(state: CollaborationState):
    plan = multi_agent_planner.invoke(f"Break this task into useful specialist tasks...\n{state['user_task']}")
    return {"tasks": plan.tasks}

def send_collaboration_workers(state: CollaborationState):
    # A conditional-edge function can return a LIST of Send objects instead of one next-node name —
    # this is how LangGraph does dynamic, data-driven fan-out (unlike Pattern 1's fixed routing table).
    return [Send("specialist_worker", {"role": t.role, "task": t.task}) for t in state["tasks"]]

def specialist_worker(state: CollaborationWorkerState):
    role_prompts = {"research": "...", "security": "...", "business": "..."}
    response = model.invoke(f"{role_prompts[state['role']]}\n\nComplete this task:\n{state['task']}")
    return {"outputs": [f"{state['role'].upper()} AGENT:\n{response.content}"]}

builder.add_conditional_edges("coordinator", send_collaboration_workers, ["specialist_worker"])
builder.add_edge("specialist_worker", "aggregator")
```

This is the multi-agent analogue of Section 20.2's subgraph **Pattern B** (`Send`-based fan-out with
an `operator.add`-reduced collection field) — same mechanism, applied to spawning parallel *agent
roles* instead of parallel *research topics*. Expected model calls: `N + 2` (1 planner + N workers +
1 aggregator).

### 23.8 Pattern 8 — Coordinator-Guided Peer Network

**Control style:** structurally identical to Supervisor (Pattern 3) — a coordinator loops, picking
the next peer until `FINISH` — but peers **share one growing `notes` scratchpad** instead of each
peer writing an isolated result tied to a specific instruction.

```text
Coordinator → Peer → Shared Notes → Coordinator → Next Peer → ... → Finish
```

```python
class NetworkState(TypedDict):
    task: str
    notes: Annotated[list[str], operator.add]     # ALL peers read + append to this shared scratchpad
    next_agent: str
    final_answer: str
    network_turns: int

def network_research_peer(state: NetworkState):
    previous = "\n\n".join(state.get("notes", []))
    response = model.invoke(f"Task:\n{state['task']}\n\nPrevious peer notes:\n{previous}\n\nAdd useful research.")
    return {"notes": ["RESEARCH:\n" + response.content]}

def peer_coordinator(state: NetworkState):
    network_turns = state.get("network_turns", 0) + 1
    notes = "\n\n".join(state.get("notes", []))
    if network_turns >= 6 and notes:                     # same safety-valve idea as Pattern 3
        return {"next_agent": "FINISH", "network_turns": network_turns}
    decision = peer_router.invoke(f"Choose the most useful next peer...\n{notes or 'No work completed yet.'}")
    return {"next_agent": decision.next_agent, "network_turns": network_turns}
```

**Supervisor vs. Peer Network, the key contrast:** same loop *shape* (`coordinator → worker →
coordinator → ... → finish`, with a turn-counter safety valve), but a different **state shape** —
Supervisor accumulates `worker_outputs` (each tied to a specific delegated instruction), while the
peer network accumulates one shared `notes` list every peer both reads and contributes to.

### 23.9 Pattern comparison table

| # | Pattern | Control mechanism | What makes it agentic (per 23.0) | Key LangGraph feature |
|---|---|---|---|---|
| 1 | Router | One decision, fixed lookup table | Router only; specialists are single-shot LLM calls | `add_conditional_edges` with a static dict |
| 2 | Subagents / Agent-as-Tool | Coordinator's own tool-calling loop | The coordinator (decides which/how many subagent tools to call) | Wrapping other agents as `@tool` functions |
| 3 | Supervisor | Central node re-decides after every worker | The supervisor (loops until FINISH) | Loop-back edges + manual turn counter |
| 4 | Handoff | One decision transfers control outright | The triage node | `Command(goto=..., update=...)` |
| 5 | Swarm | Agents decide for themselves when to transfer | Every agent (each carries its own handoff tool) | `langgraph_swarm.create_swarm` + checkpointer |
| 6 | Hierarchical Supervisor | Two-level: pick team, then fixed team pipeline | The top supervisor only | Compiled `StateGraph` invoked from a parent node |
| 7 | Parallel Collaboration (`Send`) | Dynamic, LLM-sized fan-out | The coordinator/planner | `Send` inside a conditional-edge function + `operator.add` |
| 8 | Coordinator-Guided Peer Network | Same loop as Supervisor, shared notes | The coordinator | Loop-back edges + one shared `notes` list |

**Picking a pattern:** one clear routing decision → **Router**. A main agent that should stay in
control while consulting specialists → **Subagents**. Need central oversight across several
open-ended steps → **Supervisor**. One agent should fully hand off and step away → **Handoff**.
Want agents to negotiate control without a fixed boss → **Swarm**. Work naturally splits into
departments, each with its own fixed sub-pipeline → **Hierarchical Supervisor**. Task naturally
splits into independent, parallelizable subtasks of unknown count → **Parallel Collaboration
(`Send`)**. Peers need to build on a shared scratchpad under light coordination → **Coordinator-
Guided Peer Network**.

---

## 24. Quick-reference cheat sheet

**Building any graph:**
```python
builder = StateGraph(MyState)
builder.add_node("name", node_fn)
builder.add_edge(START, "name")
builder.add_conditional_edges("name", router_fn, {"a": "node_a", "b": "node_b"})
graph = builder.compile(checkpointer=..., store=...)
graph.invoke(initial_state, config={"configurable": {"thread_id": "..."}})
```

**Node vs Tool:** Node = graph decides when it runs (deterministic). Tool = LLM decides whether it's
needed (dynamic capability).

**Tool creation:** default to `@tool`; add a Pydantic `args_schema` for validated/complex input;
make it `async` for network/DB calls; `create_retriever_tool()` for RAG; MCP for tools shared across
apps/servers.

**ReAct agent skeleton:** `llm.bind_tools(tools)` → `ToolNode(tools)` → `tools_condition` routes
between the LLM node and the tools node in a loop until no more tool calls are produced.

**Memory:**
- Short-term (per-thread) → `InMemorySaver` / `SqliteSaver` as `checkpointer=`
- Long-term (per-user, cross-thread) → `InMemoryStore` as `store=`, or a custom DB-backed store
- Compress long conversations with a sliding window and/or a running summary, not by sending
  everything every time.

**Human-in-the-loop:** `interrupt(payload)` pauses a graph (requires a checkpointer);
`graph.invoke(Command(resume=value), config=same_thread_id)` resumes it, and `interrupt()` returns
`value` at the point it paused.

**Subgraphs:** same state schema → add the compiled subgraph directly with `add_node`. Different
state schema or need N parallel copies → invoke the subgraph manually inside a worker node, fan out
with `Send`, and use an `operator.add`-reduced field to collect parallel results.

**Multi-agent pattern picker:** one clear decision → Router. Main agent stays in control, calls
specialists as tools → Subagents. Central oversight over several open-ended steps → Supervisor. One
agent fully hands off and steps away → Handoff (`Command(goto=...)`). Agents negotiate control with
no fixed boss → Swarm (`langgraph_swarm`, needs a checkpointer). Work splits into departments each
with their own sub-pipeline → Hierarchical Supervisor (graph invoked from a parent node). Unknown
number of independent parallel subtasks → Parallel Collaboration (`Send` + `operator.add`).  Peers
building on one shared scratchpad → Coordinator-Guided Peer Network. Role ≠ Agent: a node is only
truly agentic once it has its own tools and decides for itself when to stop.
