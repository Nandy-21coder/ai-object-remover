"""
app.py
Root application entrypoint for AI Object Remover.
Provides direct compatibility for `python app.py` from project root.
"""

import os
import sys
from pathlib import Path

# Ensure project root is first on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.app import app

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting AI Object Remover backend on http://{host}:{port}")
    uvicorn.run("app.backend.app:app", host=host, port=port, reload=False)
