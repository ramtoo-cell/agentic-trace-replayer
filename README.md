# Deterministic Trace Replayer for Agents

Replays recorded agent traces deterministically: seeded RNG, stubbed tool/LLM responses from the trace, and divergence detection vs the original.

**Stack:** Python 3.12 · stdlib only · unittest

## Problem statement

Given a recorded trace (spans with attributes), the replayer re-executes the agent's control flow while stubbing side effects:

## Design decisions

- `tool` spans return the recorded `result` attribute,
- `llm` spans return the recorded `completion` attribute,
- a seeded RNG reproduces the original random choices.
- `replay` walks the span tree depth-first, invoking `handlers` per span kind, and reports divergences: spans whose replayed attributes differ from the recording, or structural mismatches.

## Usage example

```python
from src.trace_recorder import TraceRecorder
from src.trace_replayer import TraceReplayer

r = TraceRecorder()
s = r.start("tool", "search", {"result": ["r1"]})
r.finish(s)
rp = TraceReplayer(r.spans, seed=1)
print(rp.replay({"tool": lambda span, ctx: {"result": ["r1"]}}))
# {'replayed': 1, 'divergences': []}
```

## Layout

- `README.md`
- `src/trace_replayer.py`
- `tests/test_trace_replayer.py`
- `LICENSE`
- `.gitignore`

## Setup

No third-party dependencies — Python 3.11+ standard library only.

```bash
cd agentic-trace-replayer
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
