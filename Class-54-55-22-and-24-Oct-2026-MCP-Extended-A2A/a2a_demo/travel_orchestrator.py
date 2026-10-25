# =============================================================================
# TRAVEL ORCHESTRATOR  (an A2A "client" agent)
# =============================================================================
# BIG PICTURE
# An "orchestrator" is a manager agent. It doesn't do the specialised work
# itself; it finds other agents, delegates tasks to them, and gathers results.
#
#   You (user request)
#          |
#          v
#   Travel Orchestrator  --asks-->  Flight Agent  (http://127.0.0.1:9999)
#          |
#          +-------------asks-->  Hotel Agent   (http://127.0.0.1:10000)
#
# For each remote agent the orchestrator does 3 things:
#   1. DISCOVER : download the agent's "Agent Card" (its profile) from its URL.
#   2. CONNECT  : create an A2A client using the info on that card.
#   3. ASK      : send a text message and read the reply.
#
# HOW TO RUN: start flight_agent.py and hotel_agent.py FIRST (each in its own
# terminal), then run:  python travel_orchestrator.py
# =============================================================================

# asyncio lets us run "async" functions (ones that wait on network calls
# without freezing the program).
import asyncio

# httpx is a Python library for making HTTP requests (like a code-based browser).
import httpx

# A2ACardResolver : fetches an agent's card from its base URL (discovery).
# ClientConfig    : settings for the client (e.g. streaming on/off).
# create_client   : builds a ready-to-use client that can talk to an agent.
from a2a.client import A2ACardResolver, ClientConfig, create_client
# Same helpers as in the agents: read text from / build text messages.
from a2a.helpers import get_message_text, new_text_message
# Message            : the A2A message object type.
# Role               : who is speaking (user or agent).
# SendMessageRequest : the request object we send to an agent.
from a2a.types import Message, Role, SendMessageRequest


# -----------------------------------------------------------------------------
# HELPER FUNCTION: talk to ONE remote agent
# -----------------------------------------------------------------------------
# Give it the agent's address (base_url) and what you want done (task_text);
# it returns the agent's text reply. We reuse it for both Flight and Hotel.
# "async def" = this function can use "await" to wait for network responses.
async def call_remote_agent(
    base_url: str,
    task_text: str,
) -> str:
    # Open an HTTP connection session. "async with" guarantees it is closed
    # properly when the block ends, even if an error happens.
    async with httpx.AsyncClient() as httpx_client:
        # STEP 1 - DISCOVERY: the resolver knows where agents publish their
        # cards (the standard path /.well-known/agent-card.json) and fetches it.
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Download the agent card. "await" = pause here until it arrives.
        agent_card = await resolver.get_agent_card()

        # Print what we learned about the remote agent (its name and skills)
        # so you can SEE discovery happening.
        print(f"\nDiscovered Agent: {agent_card.name}")
        print("Skills:")
        for skill in agent_card.skills:
            print("-", skill.name)

        # STEP 2 - CONNECT: build a client from the agent card. The card tells
        # it the URL and protocol (JSON-RPC) to use, so we don't hard-code them.
        client = await create_client(
            agent=agent_card,
            client_config=ClientConfig(
                streaming=False   # wait for one full reply instead of a live stream
            ),
        )

        # "try ... finally" makes sure client.close() ALWAYS runs at the end,
        # even if something fails in the middle (like closing a door behind you).
        try:
            # STEP 3 - ASK: wrap our task text in an A2A message.
            # Role.ROLE_USER marks the sender as the "user" side of the chat.
            message = new_text_message(
                task_text,
                role=Role.ROLE_USER,
            )

            # Put the message inside a request object, ready to send.
            request = SendMessageRequest(
                message=message
            )

            # Send the request and read the response(s). send_message yields
            # results one at a time, so we loop with "async for".
            async for response in client.send_message(request):
                # If the reply is a plain Message (what our demo agents send),
                # extract and return its text.
                if isinstance(response, Message):
                    return get_message_text(response)

                # Otherwise (e.g. a Task object) just convert it to a string.
                # Either way we return after the first response.
                return str(response)

        finally:
            # Always release the client's resources.
            await client.close()

    # Safety net: only reached if the agent sent no response at all.
    return "No response received."


# -----------------------------------------------------------------------------
# MAIN FLOW: the orchestrator's "plan"
# -----------------------------------------------------------------------------
async def main():
    # Print a banner. "=" * 65 repeats the "=" character 65 times.
    print("=" * 65)
    print("TRAVEL ORCHESTRATOR AGENT")
    print("=" * 65)

    # The big request from the (pretend) user. It needs TWO specialists:
    # flights and a hotel. (Adjacent strings are joined into one.)
    user_request = (
        "Plan a 5-day Dubai trip from Bangalore. "
        "I need a flight and a 4-star hotel."
    )

    print("\nUSER REQUEST:")
    print(user_request)

    # DELEGATION 1: split the big request and send the flight part to the
    # Flight Agent (port 9999). "await" waits for its answer.
    # Note: here the orchestrator hard-codes the split. In a real system an
    # LLM would decide which agent to call and what to ask.
    flight_result = await call_remote_agent(
        "http://127.0.0.1:9999",
        "Find good flights from Bangalore to Dubai.",
    )

    # DELEGATION 2: send the hotel part to the Hotel Agent (port 10000).
    hotel_result = await call_remote_agent(
        "http://127.0.0.1:10000",
        "Find a 4-star hotel in Dubai for 5 nights.",
    )

    # Show what each agent replied, under clear headings.
    print("\n" + "=" * 65)
    print("FLIGHT AGENT RESPONSE")
    print("=" * 65)
    print(flight_result)

    print("\n" + "=" * 65)
    print("HOTEL AGENT RESPONSE")
    print("=" * 65)
    print(hotel_result)

    # Final summary. Here it's just a fixed message; a real orchestrator
    # would combine both results into one itinerary (maybe using an LLM).
    print("\n" + "=" * 65)
    print("FINAL ORCHESTRATED RESULT")
    print("=" * 65)
    print(
        "Travel Orchestrator delegated work to two independent "
        "A2A agents and collected their responses."
    )


# Run main() only when this file is executed directly. asyncio.run() starts
# the async event loop that is needed to run an "async def" function.
if __name__ == "__main__":
    asyncio.run(main())
