# =============================================================================
# HOTEL AGENT  (an A2A "server" agent)
# =============================================================================
# BIG PICTURE
# A2A = "Agent-to-Agent" protocol. It lets independent AI agents talk to each
# other over HTTP, even if they are built by different teams or frameworks.
#
# This file is almost identical to flight_agent.py. Only the details differ
# (name, skill, fake data, port number). If you understood the Flight Agent,
# you already understand this one! See flight_agent.py for the longest
# explanations of each concept.
#
# THIS FILE is the Hotel Agent. It:
#   - describes itself with an "Agent Card" (like a business card / profile),
#   - listens for incoming messages over HTTP (JSON-RPC) on port 10000,
#   - replies with (hard-coded, fake) hotel results.
#
# HOW TO RUN: python hotel_agent.py   (keep it running in its own terminal)
# =============================================================================

# uvicorn: the web server that listens on a network port and serves our app.
import uvicorn

# get_message_text : extract plain text from an A2A message.
# new_text_message : build an A2A message from a plain string.
from a2a.helpers import get_message_text, new_text_message
# AgentExecutor  : base class for our agent's logic.
# RequestContext : info about the incoming request (including the user's message).
from a2a.server.agent_execution import AgentExecutor, RequestContext
# EventQueue: a "mailbox" where the agent puts replies to be delivered to the caller.
from a2a.server.events import EventQueue
# DefaultRequestHandler: framework glue that receives requests and runs our executor.
from a2a.server.request_handlers import DefaultRequestHandler
# Helpers that create web URLs for the agent card (discovery) and JSON-RPC (messaging).
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
# InMemoryTaskStore: remembers tasks in RAM only (lost when the server stops).
from a2a.server.tasks import InMemoryTaskStore
# Classes used to describe the agent publicly (profile, skills, capabilities, address).
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
# Starlette: a lightweight Python web framework used to build the web app.
from starlette.applications import Starlette


# -----------------------------------------------------------------------------
# STEP 1: THE AGENT'S BRAIN (the "Executor")
# -----------------------------------------------------------------------------
# Holds the logic that runs whenever someone sends this agent a message.
class HotelAgentExecutor(AgentExecutor):
    # execute() runs automatically each time a message arrives.
    #   context     -> the incoming request (has the caller's message)
    #   event_queue -> where we place our reply so it is sent back
    # "async"/"await" let Python wait for slow things (like network calls)
    # without freezing the whole program.
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Get the caller's text; use "" if there's no message to avoid a crash.
        query = get_message_text(context.message) if context.message else ""

        # Fake, hard-coded hotel options. A real agent might query a booking
        # API or ask an LLM. We also echo the received request for visibility.
        # (Neighbouring strings are joined into one; "\n" = new line.)
        result = (
            "Hotel Agent result:\n"
            "Dubai, 5 nights\n"
            "1. Business Bay Hotel | 4-star | ₹9,500/night\n"
            "2. Downtown Executive Hotel | 4-star | ₹11,000/night\n"
            f"\nReceived task: {query}"
        )

        # Wrap the text in an A2A message and put it in the queue; the
        # framework delivers it back to the caller.
        await event_queue.enqueue_event(
            new_text_message(result)
        )

    # cancel() would stop a running task. Our agent replies instantly, so
    # there is nothing to cancel and we say it is not supported.
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Cancellation is not supported.")


# -----------------------------------------------------------------------------
# STEP 2: DESCRIBE WHAT THE AGENT CAN DO (the "Skill")
# -----------------------------------------------------------------------------
# A skill is one advertised capability so others can decide if this is the
# right agent to ask.
skill = AgentSkill(
    id="find_hotels",                         # unique machine-friendly ID
    name="Find Hotels",                       # human-friendly name
    description="Finds hotel options for a destination.",
    input_modes=["text/plain"],               # accepts plain text
    output_modes=["text/plain"],              # returns plain text
    tags=["travel", "hotel"],                 # keywords to help discovery
    examples=["Find a 4-star hotel in Dubai"],  # sample requests
)

# -----------------------------------------------------------------------------
# STEP 3: THE AGENT CARD (the agent's "business card")
# -----------------------------------------------------------------------------
# A public JSON profile that clients download to learn who this agent is,
# what it can do, and how/where to talk to it. This is "agent discovery".
agent_card = AgentCard(
    name="Hotel Agent",
    description="Specialized agent for finding hotels.",
    version="1.0.0",
    # Default input/output types when a skill doesn't override them.
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(
        streaming=False,            # one full reply, not a live stream of chunks
        extended_agent_card=False,  # no second, more detailed card offered
    ),
    # HOW to reach the agent: JSON-RPC over HTTP at this address.
    # NOTE the port is 10000 here (the Flight Agent uses 9999), so the two
    # agents can run side by side on the same computer.
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="http://127.0.0.1:10000",   # 127.0.0.1 = localhost (this computer)
            protocol_version="1.0",
        )
    ],
    skills=[skill],                 # skills this agent offers
)

# -----------------------------------------------------------------------------
# STEP 4: CONNECT EVERYTHING TOGETHER (the "Request Handler")
# -----------------------------------------------------------------------------
# The traffic controller: gives incoming requests to our executor, tracks the
# task, and returns the response.
request_handler = DefaultRequestHandler(
    agent_executor=HotelAgentExecutor(),   # our logic from STEP 1
    task_store=InMemoryTaskStore(),        # task storage (RAM only)
    agent_card=agent_card,                 # our profile from STEP 3
)

# -----------------------------------------------------------------------------
# STEP 5: CREATE THE WEB APP AND ITS URLS (routes)
# -----------------------------------------------------------------------------
# Two groups of routes:
#   1. Agent card routes -> clients download our card (usually at
#                           /.well-known/agent-card.json).
#   2. JSON-RPC routes   -> "/" is where clients send us messages.
# The "*" unpacks each list into one flat list of routes.
app = Starlette(
    routes=[
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(request_handler, "/"),
    ]
)

# -----------------------------------------------------------------------------
# STEP 6: START THE SERVER
# -----------------------------------------------------------------------------
# Runs only when you execute this file directly (python hotel_agent.py),
# not when it is imported. Starts the server at 127.0.0.1:10000, which must
# match the URL in the agent card and the one the orchestrator calls.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=10000)
