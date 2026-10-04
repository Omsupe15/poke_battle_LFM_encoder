import os
import socket
import subprocess
import sys
import time
from typing import Optional


def is_server_running(host: str = "localhost", port: int = 8000) -> bool:
    """Checks whether the Pokémon Showdown server is listening on the given host/port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def get_showdown_repo_dir() -> str:
    """Locates the local pokemon-showdown repository directory."""
    # Look in the current working directory or relative to this script
    candidates = [
        os.path.join(os.getcwd(), "pokemon-showdown"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pokemon-showdown")),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "pokemon-showdown")):
            return path
    raise FileNotFoundError(
        "Could not locate the 'pokemon-showdown' directory. Ensure it is cloned in your workspace."
    )


def start_local_showdown(port: int = 8000, timeout_seconds: int = 15) -> subprocess.Popen:
    """
    Starts a local Pokémon Showdown server in the background with --no-security
    if not already running.
    """
    if is_server_running(port=port):
        print(f"[INFO] Pokémon Showdown server is already running on port {port}.")
        return None

    showdown_dir = get_showdown_repo_dir()
    print(f"[INFO] Starting local Pokémon Showdown from {showdown_dir} on port {port}...")

    cmd = ["node", "pokemon-showdown", "start", str(port), "--no-security"]
    process = subprocess.Popen(
        cmd,
        cwd=showdown_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=sys.platform == "win32"
    )

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if is_server_running(port=port):
            print(f"[SUCCESS] Pokémon Showdown server is online at port {port}.")
            return process
        time.sleep(0.5)

    raise TimeoutError(f"Failed to connect to Pokémon Showdown on port {port} after {timeout_seconds}s.")
