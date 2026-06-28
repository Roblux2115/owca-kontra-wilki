import json
import time
import urllib.error
import urllib.request


STATE_API_URL = "http://127.0.0.1:8000"


class StateReporter:
    def __init__(self, agent_name, enabled=True):
        self.agent_name = agent_name
        self.enabled = enabled

    def update(self, status=None, **metrics):
        if not self.enabled:
            return

        payload = {
            "agent": self.agent_name,
            "status": status,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        _post_json("/state/update", payload)

    def event(self, event_type, **data):
        if not self.enabled:
            return

        payload = {
            "agent": self.agent_name,
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        _post_json("/events", payload)

    def config(self, **config):
        if not self.enabled:
            return

        payload = {
            "agent": self.agent_name,
            "config": config,
        }
        _post_json("/config", payload)


def _post_json(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        STATE_API_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=0.35).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
