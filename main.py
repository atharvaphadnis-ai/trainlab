"""TrainLab entry point.

 python main.py

What it does:
Checks that the needed packages are installed.
Starts the FastAPI server on 127.0.0.1 (port from settings.json).
Opens the TrainLab website in the default browser.
 Shuts down cleanly on Ctrl+C or when the console window is closed.


"""

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from utils import APP_NAME, app_base_dir, load_settings

REQUIRED_PACKAGES = ["fastapi", "uvicorn", "PIL", "numpy", "multipart"]


def check_required_packages():
    """Exit with a helpful message when a package is missing."""
    import importlib

    missing = []
    for package_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print("Some required packages are missing:", ", ".join(missing))
        print("Install everything with:  pip install -r requirements.txt")
        print("TensorFlow is checked later, when you first train a model.")
        sys.exit(1)


def index_html_exists():
    """Make sure index.html was bundled correctly (PyInstaller --add-data)."""
    index_file = app_base_dir() / "index.html"
    if not index_file.exists():
        print(f"index.html was not found next to {Path(sys.argv[0]).name}.")
        print("Run the app from the TrainLab folder, or rebuild the exe with:")
        print('  pyinstaller --name TrainLab --onefile --add-data "index.html;." main.py')
        return False
    return True


def port_is_busy(port):
    """True when something already listens on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def wait_until_server_ready(server, timeout_seconds=60):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server.started:
            return True
        time.sleep(0.2)
    return False


def run_server():
    settings = load_settings()
    port = int(settings["port"])
    url = f"http://127.0.0.1:{port}"

    if port_is_busy(port):
        # Another TrainLab (or another app) already uses this port.
        # Opening a second server would be wrong, so just open the browser.
        print(f"Port {port} is already in use.")
        print(f"If TrainLab is already running, it should be here: {url}")
        webbrowser.open(url)
        return

    import uvicorn

    import server  # imports the routes and creates the database

    server_config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="info")
    server_instance = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server_instance.run, daemon=True)
    server_thread.start()

    if not wait_until_server_ready(server_instance):
        print("The server did not start. Check the messages above for errors.")
        return

    print(f"{APP_NAME} is running at {url}")
    print("Keep this window open. Press Ctrl+C to stop TrainLab.")
    webbrowser.open(url)

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping TrainLab...")
        server_instance.should_exit = True
        server_thread.join(timeout=10)
        print("TrainLab has stopped.")


def main():
    check_required_packages()
    if not index_html_exists():
        sys.exit(1)
    run_server()


if __name__ == "__main__":
    main()
