import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.storage import read_json, write_json


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
@pytest.mark.parametrize("filename", ["summary.json", "result.json.gz"])
def test_json_replacement_waits_for_a_live_windows_reader(tmp_path, filename):
    path = tmp_path / filename
    write_json(path, {"runs": 125})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create.restype = ctypes.c_void_p
    close = kernel.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    handle = create(str(path), 0x80000000, 0x00000003, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    started = threading.Event()

    def publish():
        started.set()
        write_json(path, {"runs": 160})

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(publish)
        try:
            assert started.wait(2)
            time.sleep(0.12)
            assert not future.done()
            assert read_json(path) == {"runs": 125}
        finally:
            assert close(handle)
        future.result(timeout=3)
    assert read_json(path) == {"runs": 160}
    assert not path.with_suffix(".tmp").exists()


def test_persistent_sharing_failure_raises_and_preserves_old_json(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    write_json(path, {"runs": 125})
    calls = []

    def locked_replace(source, destination):
        calls.append((source, destination))
        error = PermissionError("still open")
        error.winerror = 5
        raise error

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("experiments.storage.time.sleep", lambda _: None)
    with pytest.raises(PermissionError, match="still open"):
        write_json(path, {"runs": 160})
    assert len(calls) == 20
    assert read_json(path) == {"runs": 125}
    assert read_json(path.with_suffix(".tmp")) == {"runs": 160}
