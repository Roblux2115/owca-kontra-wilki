import json
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY_PREFIX = "owca_kontra_wilki"

app = FastAPI(title="Owca Kontra Wilki - State API")


class StateUpdate(BaseModel):
    agent: str = Field(min_length=1)
    status: str | None = None
    metrics: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class EventUpdate(BaseModel):
    agent: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    data: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class ConfigUpdate(BaseModel):
    agent: str = Field(min_length=1)
    config: dict = Field(default_factory=dict)


def redis_client():
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def key(*parts):
    return ":".join([KEY_PREFIX, *parts])


@app.get("/health")
def health():
    client = redis_client()
    try:
        client.ping()
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc
    return {"status": "ok", "redis_url": REDIS_URL}


@app.post("/state/update")
def update_state(update: StateUpdate):
    client = redis_client()
    state_key = key("agent", update.agent, "state")
    current = _read_json(client, state_key, {})
    current.update(update.metrics)
    if update.status is not None:
        current["status"] = update.status
    current["updated_at"] = update.timestamp
    client.set(state_key, json.dumps(current, ensure_ascii=False))
    return {"saved": True, "agent": update.agent, "state": current}


@app.post("/events")
def append_event(event: EventUpdate):
    client = redis_client()
    event_data = event.dict()
    client.lpush(key("events", event.agent), json.dumps(event_data, ensure_ascii=False))
    client.ltrim(key("events", event.agent), 0, 49)
    return {"saved": True}


@app.post("/config")
def save_config(update: ConfigUpdate):
    client = redis_client()
    client.set(key("agent", update.agent, "config"), json.dumps(update.config, ensure_ascii=False))
    return {"saved": True, "agent": update.agent, "config": update.config}


@app.get("/state")
def read_all_state():
    client = redis_client()
    agents = {}
    for state_key in client.scan_iter(key("agent", "*", "state")):
        agent = state_key.split(":")[-2]
        agents[agent] = _read_json(client, state_key, {})
    return {"agents": agents}


@app.get("/state/{agent}")
def read_agent_state(agent: str):
    client = redis_client()
    state = _read_json(client, key("agent", agent, "state"), {})
    config = _read_json(client, key("agent", agent, "config"), {})
    events = [
        json.loads(item)
        for item in client.lrange(key("events", agent), 0, 9)
    ]
    return {"agent": agent, "state": state, "config": config, "recent_events": events}


def _read_json(client, redis_key, default):
    raw = client.get(redis_key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default
