"""Frozen parameter checks and one-observation oracle IPC."""
import hashlib

from .data import Observation


def tensor_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class RemoteOracle:
    def __init__(self, connection):
        self.connection = connection

    def query(self, index):
        self.connection.send({"type": "query", "index": int(index)})
        value = self.connection.recv()
        if value.get("type") == "error":
            raise RuntimeError(value["message"])
        return Observation(**value["observation"])
