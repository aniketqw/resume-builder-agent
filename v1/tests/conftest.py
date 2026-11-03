import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The suite runs offline. Tracing would otherwise try to reach LangSmith on
# every fake LLM call, which turned a 0.8s run into a 5.8s one and makes the
# tests depend on a network that CI should not need.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_TRACING_V2", None)
