#!/usr/bin/env python3
"""
magicpin AI Challenge — Multi-turn Conversation Handlers ("conversation_handlers.py")
====================================================================================

Demonstrates multi-turn handling (replying to merchant / customer responses).
Imported or called by judge during multi-turn replays.
"""

from bot import respond as bot_respond

def respond(state: dict, merchant_message: str) -> dict:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.
    Inputs:
        state: dict containing conversation history, merchant_id, turn_number
        merchant_message: string message from merchant or customer
    Returns:
        dict with action ("send" | "wait" | "end"), body (if send), cta, rationale
    """
    return bot_respond(state, merchant_message)
