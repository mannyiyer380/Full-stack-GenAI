# =============================================================================
# FLIGHT AGENT  (an A2A "server" agent)
# =============================================================================
# BIG PICTURE
# A2A = "Agent-to-Agent" protocol. It lets independent AI agents talk to each
# other over HTTP, even if they are built by different teams or frameworks.
#
# In this demo there are 3 files:
#   1. flight_agent.py        -> a small web server that "knows" about flights  (port 9999)
#   2. hotel_agent.py         -> a small web server that "knows" about hotels   (port 10000)
#   3. travel_orchestrator.py -> a client that discovers both agents and asks
#                                each one to do part of a trip-planning job.
#
# THIS FILE is the Flight Agent. It:
#   - describes itself with an "Agent Card" (like a business card / profile),
#   - listens for incoming messages over HTTP (JSON-RPC),
#   - replies with (hard-coded, fake) flight results.
#
# HOW TO RUN (each in its own terminal, agents first, orchestrator last):
#   python flight_agent.py
#   python hotel_agent.py
#   python travel_orchestrator.py
# =============================================================================

# uvicorn is the web server that actually listens on a network port and
# runs our app. Think of it as the "engine" that serves our agent over HTTP.
import uvicorn

# get_message_text : pulls the plain text out of an A2A message object.
# new_text_message : builds a new A2A message object from a plain string.
from a2a.helpers import get_message_text, new_text_message
# AgentExecutor  : the base class we extend to write our agent's actual logic.
# RequestContext : holds everything about the incoming request (e.g. the user's message).
from a2a.server.agent_execution import AgentExecutor, RequestContext
# EventQueue : a "mailbox" where the agent drops its replies/events so the
#              framework can deliver them back to the caller.
from a2a.server.events import EventQueue
# DefaultRequestHandler : the built-in glue that receives requests, calls our
#                         executor, and sends the results back.
from a2a.server.request_handlers import DefaultRequestHandler
# These helpers create the web URLs (routes) for:
#   - the agent card (so others can discover us), and
#   - the JSON-RPC endpoint (where others send us messages).
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
# InMemoryTaskStore : remembers tasks while the program runs. "In memory" means
#                     everything is lost when the server stops (fine for a demo).
from a2a.server.tasks import InMemoryTaskStore
# Types used to describe the agent:
#   AgentCard         - the agent's public profile
#   AgentSkill        - one thing the agent can do
#   AgentCapabilities - optional features the agent supports (streaming etc.)
#   AgentInterface    - HOW and WHERE to reach the agent (protocol + URL)
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
# Starlette is a lightweight Python web framework. We use it to build the web app.
from starlette.applications import Starlette


# -----------------------------------------------------------------------------
# STEP 1: THE AGENT'S BRAIN (the "Executor")
# -----------------------------------------------------------------------------
# An Executor holds the logic that runs when someone sends this agent a message.
# We inherit from AgentExecutor and must implement two methods:
# execute() and cancel().
class FlightAgentExecutor(AgentExecutor):
    # "async" means this function can pause while waiting (e.g. for network I/O)
    # without blocking the whole program. Python's "await" is used inside it.
    #
    # execute() is called automatically every time a message arrives.
    #   context     -> the incoming request (contains the caller's message)
    #   event_queue -> where we put our answer so it gets sent back
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Read the text the caller sent us. If for some reason there is no
        # message, fall back to an empty string so the code doesn't crash.
        query = get_message_text(context.message) if context.message else ""

        # In a real agent you would call an airline API or an LLM here.
        # For this demo we simply return fixed, made-up flight options and
        # echo back the request we received so you can see it worked.
        # (Adjacent strings in parentheses are joined into ONE string;
        #  "\n" means "new line"; the f"..." lets us insert {query}.)
        result = (
            "Flight Agent result:\n"
            "Bangalore -> Dubai\n"
            "1. Emirates EK569 | Non-stop | ₹28,000\n"
            "2. Air India AI995 | 1 stop | ₹24,500\n"
            f"\nReceived task: {query}"
        )

        # Wrap our text in an A2A message and put it on the event queue.
        # The framework picks it up and sends it back to whoever called us.
        await event_queue.enqueue_event(
            new_text_message(result)
        )

    # cancel() is called if the caller wants to stop a running task.
    # Our demo agent answers instantly, so there is nothing to cancel;
    # we just raise an error saying cancellation isn't supported.
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Cancellation is not supported.")


# -----------------------------------------------------------------------------
# STEP 2: DESCRIBE WHAT THE AGENT CAN DO (the "Skill")
# -----------------------------------------------------------------------------
# A skill is one capability, advertised publicly so other agents (or an LLM)
# can decide whether this agent is the right one to ask.
skill = AgentSkill(
    id="search_flights",                      # unique machine-friendly ID
    name="Search Flights",                    # human-friendly name
    description="Searches flight options between cities.",
    input_modes=["text/plain"],               # kinds of input accepted (plain text)
    output_modes=["text/plain"],              # kinds of output produced (plain text)
    tags=["travel", "flight"],                # keywords to help with searching/discovery
    examples=["Find flights from Bangalore to Dubai"],  # sample requests
)

# -----------------------------------------------------------------------------
# STEP 3: THE AGENT CARD (the agent's "business card")
# -----------------------------------------------------------------------------
# The Agent Card is a JSON document that the server publishes. Any client can
# download it to learn: who is this agent, what can it do, and how do I talk
# to it? This is how "agent discovery" works in A2A.
agent_card = AgentCard(
    name="Flight Agent",
    description="Specialized agent for searching flights.",
    version="1.0.0",
    # Defaults used when a skill doesn't specify its own input/output types.
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(
        streaming=False,            # we send one full reply, not a live stream of chunks
        extended_agent_card=False,  # we don't offer a second, more detailed card
    ),
    # HOW to reach this agent: speak JSON-RPC (a simple request/response
    # format over HTTP) at this URL. Port 9999 is where we start the server below.
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="http://127.0.0.1:9999",   # 127.0.0.1 = "this computer" (localhost)
            protocol_version="1.0",
        )
    ],
    skills=[skill],                 # list of skills this agent offers
)

# -----------------------------------------------------------------------------
# STEP 4: CONNECT EVERYTHING TOGETHER (the "Request Handler")
# -----------------------------------------------------------------------------
# The request handler is the traffic controller: when a request arrives it
# hands it to our executor, tracks the task, and returns the response.
request_handler = DefaultRequestHandler(
    agent_executor=FlightAgentExecutor(),  # our logic from STEP 1
    task_store=InMemoryTaskStore(),        # where task info is kept (RAM only)
    agent_card=agent_card,                 # our profile from STEP 3
)

# -----------------------------------------------------------------------------
# STEP 5: CREATE THE WEB APP AND ITS URLS (routes)
# -----------------------------------------------------------------------------
# A "route" maps a URL to some code. We register two groups of routes:
#   1. Agent card routes  -> lets clients download the card (discovery).
#                            By A2A convention it lives at /.well-known/agent-card.json
#   2. JSON-RPC routes    -> the "/" URL where clients send us messages.
# The "*" unpacks each list so all routes sit together in one flat list.
app = Starlette(
    routes=[
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(request_handler, "/"),
    ]
)

# -----------------------------------------------------------------------------
# STEP 6: START THE SERVER
# -----------------------------------------------------------------------------
# This block only runs when you execute this file directly
# (python flight_agent.py), not when another file imports it.
# It starts the web server on this computer (127.0.0.1) at port 9999.
# The port must match the URL in the agent card above and the one the
# orchestrator calls.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)
