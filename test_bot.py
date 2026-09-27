#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Vera AI Assistant (bot.py)
"""

import pytest
from fastapi.testclient import TestClient
from bot import app, compose, respond, clear_state, CONTEXT_STORE, SUPPRESSED_KEYS

client = TestClient(app)

def setup_function():
    clear_state()

def test_healthz_initial():
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}

def test_metadata():
    response = client.get("/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "team_name" in data
    assert "version" in data

def test_context_push_and_idempotency():
    payload = {
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": {"slug": "dentists", "voice": {"tone": "peer_clinical"}}
    }
    # Push version 1
    res1 = client.post("/v1/context", json=payload)
    assert res1.status_code == 200
    assert res1.json()["accepted"] is True

    # Re-push version 1 (should return 409 conflict)
    res2 = client.post("/v1/context", json=payload)
    assert res2.status_code == 409
    assert res2.json()["accepted"] is False
    assert res2.json()["reason"] == "stale_version"

    # Push version 2 (should accept)
    payload["version"] = 2
    res3 = client.post("/v1/context", json=payload)
    assert res3.status_code == 200
    assert res3.json()["accepted"] is True

def test_tick_and_reply_flow():
    # Push Category
    client.post("/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
    })
    # Push Merchant
    client.post("/v1/context", json={
        "scope": "merchant", "context_id": "m_001_drmeera", "version": 1,
        "payload": {
            "merchant_id": "m_001_drmeera", "category_slug": "dentists",
            "identity": {"name": "Dr. Meera's Dental Clinic", "city": "Delhi", "locality": "Lajpat Nagar", "owner_first_name": "Meera", "languages": ["en", "hi"]},
            "performance": {"views": 2410, "calls": 18, "ctr": 0.021},
            "offers": [{"id": "o1", "title": "Dental Cleaning @ ₹299", "status": "active"}]
        }
    })
    # Push Trigger
    client.post("/v1/context", json={
        "scope": "trigger", "context_id": "trg_001", "version": 1,
        "payload": {
            "id": "trg_001", "scope": "merchant", "kind": "perf_dip",
            "merchant_id": "m_001_drmeera", "urgency": 4, "suppression_key": "perf_dip:m_001:2026",
            "payload": {"metric": "calls", "delta_pct": -0.50, "vs_baseline": 12}
        }
    })

    # Call /v1/tick
    tick_res = client.post("/v1/tick", json={"now": "2026-04-26T10:00:00Z", "available_triggers": ["trg_001"]})
    assert tick_res.status_code == 200
    actions = tick_res.json()["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["send_as"] == "vera"
    assert "calls" in action["body"].lower() or "drop" in action["body"].lower() or "50%" in action["body"]
    assert action["cta"] == "binary_yes_no"

    # Call /v1/reply with intent transition
    conv_id = action["conversation_id"]
    reply_res = client.post("/v1/reply", json={
        "conversation_id": conv_id, "merchant_id": "m_001_drmeera",
        "from_role": "merchant", "message": "Yes let's do it. What's next?", "turn_number": 2
    })
    assert reply_res.status_code == 200
    reply_data = reply_res.json()
    assert reply_data["action"] == "send"
    assert "drafted" in reply_data["body"].lower() or "done" in reply_data["body"].lower()

def test_auto_reply_detection():
    auto_msg = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."
    state = {"turns": []}

    res1 = respond(state, auto_msg)
    assert res1["action"] == "wait"
    assert res1["wait_seconds"] == 14400

    state["turns"].append({"from": "merchant", "msg": auto_msg})
    res2 = respond(state, auto_msg)
    assert res2["action"] == "end"

def test_hostile_handling():
    res = respond({"turns": []}, "Stop messaging me. This is useless spam.")
    assert res["action"] == "end"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
