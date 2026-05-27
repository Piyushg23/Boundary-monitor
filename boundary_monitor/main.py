"""
main.py — Entry point for Boundary Monitor v3.
"""

# config.py loads .env automatically at import time, so this is all we need.
if __name__ == "__main__":
    from ui.launcher import launch_gui
    launch_gui()
