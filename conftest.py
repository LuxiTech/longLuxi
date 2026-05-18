"""Root conftest: add project root to sys.path so top-level packages like
``eval`` and ``pipeline`` are importable in tests without being installed."""
import sys
from pathlib import Path

# Insert the project root (parent of this file) at the front of sys.path.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
