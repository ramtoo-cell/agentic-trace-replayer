'''Tests for the deterministic trace replayer.'''

import unittest

from types import SimpleNamespace

from src.trace_replayer import Divergence, TraceReplayer


def _span(sid, parent_id, kind, name, attributes=None):
    return SimpleNamespace(id=sid, trace_id="t-1", parent_id=parent_id,
                           kind=kind, name=name,
                           attributes=dict(attributes or {}))


def recorded_trace():
    return [
        _span(1, None, "agent", "run"),
        _span(2, 1, "tool", "search", {"q": "ai", "result": ["r1"]}),
        _span(3, 1, "llm", "summarize",
              {"completion": "summary text"}),
        _span(4, 1, "eval", "check", {"score": 0.9}),
    ]


def handlers():
    return {
        "agent": lambda span, ctx: {},
        "tool": lambda span, ctx: {"result": span.attributes["result"]},
        "llm": lambda span, ctx: {"completion": span.attributes["completion"]},
    }


class TraceReplayerTests(unittest.TestCase):
    def test_replay_no_divergence(self):
        rp = TraceReplayer(recorded_trace(), seed=42)
        report = rp.replay(handlers())
        self.assertEqual(report["replayed"], 4)

    def test_divergence_detected(self):
        rp = TraceReplayer(recorded_trace())

        def bad_tool(span, ctx):
            return {"result": ["DIFFERENT"]}

        report = rp.replay({"agent": lambda s, c: {}, "tool": bad_tool,
                            "llm": lambda s, c: {}})
        self.assertEqual(len(report["divergences"]), 1)
        self.assertEqual(report["divergences"][0]["span"], 2)

    def test_strict_raises(self):
        rp = TraceReplayer(recorded_trace(), strict=True)

        def bad_tool(span, ctx):
            return {"result": ["DIFFERENT"]}

        with self.assertRaises(Divergence):
            rp.replay({"tool": bad_tool})

    def test_seeded_rng_reproducible(self):
        spans = recorded_trace()
        seen = []
        h = {"agent": lambda s, ctx: seen.append(ctx["rng"].random()) or {}}
        TraceReplayer(spans, seed=7).replay(h)
        TraceReplayer(spans, seed=7).replay(h)
        self.assertEqual(seen[0], seen[1])

    def test_missing_handler_ok(self):
        rp = TraceReplayer(recorded_trace())
        report = rp.replay({})
        self.assertEqual(report["replayed"], 4)

    def test_stub_tool(self):
        rp = TraceReplayer(recorded_trace())
        stubs = rp.stub_tool("search", ["cached"])
        self.assertEqual(stubs["search"], ["cached"])

    def test_empty_trace(self):
        report = TraceReplayer([]).replay({})
        self.assertEqual(report["replayed"], 0)

    def test_replay_resets_state(self):
        rp = TraceReplayer(recorded_trace())
        rp.replay(handlers())
        report = rp.replay(handlers())
        self.assertEqual(report["replayed"], 4)
        self.assertEqual(len(rp.replayed), 4)

from src.trace_replayer import TraceReplayer


class TraceReplayerExtraTests(unittest.TestCase):
    def test_kinds_and_coverage(self):
        rp = TraceReplayer(recorded_trace())
        self.assertEqual(rp.kinds(), ["agent", "eval", "llm", "tool"])
        self.assertEqual(rp.coverage(handlers()), 0.75)
        self.assertEqual(rp.coverage({}), 0.0)
        self.assertEqual(rp.span(2).name, "search")




if __name__ == "__main__":
    unittest.main()
