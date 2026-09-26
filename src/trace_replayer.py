'''Deterministic replayer for recorded agent traces.

Given a recorded trace (spans with attributes), the replayer
re-executes the agent's control flow while stubbing side effects:

* ``tool`` spans return the recorded ``result`` attribute,
* ``llm`` spans return the recorded ``completion`` attribute,
* a seeded RNG reproduces the original random choices.

``replay`` walks the span tree depth-first, invoking ``handlers``
per span kind, and reports divergences: spans whose replayed attributes
differ from the recording, or structural mismatches.  Spans are plain
data (any object with ``id``/``parent_id``/``kind``/``attributes``);
no recorder dependency is needed.
'''

from __future__ import annotations

import random


class Divergence(Exception):
    '''Raised in strict mode on replay divergence.'''


class TraceReplayer:
    '''Replays span trees with stubbed side effects.'''

    def __init__(self, spans, seed=0, strict=False):
        '''spans: list of Span-like objects with id/parent_id/kind/name.'''
        self.spans = list(spans)
        self.seed = seed
        self.strict = strict
        self.rng = random.Random(seed)
        self.divergences = []
        self.replayed = []

    def _children(self, span_id):
        return sorted(
            (s for s in self.spans if s.parent_id == span_id),
            key=lambda s: s.id)

    def span(self, span_id):
        for span in self.spans:
            if span.id == span_id:
                return span
        raise KeyError(f"unknown span: {span_id}")

    def kinds(self):
        '''Span kinds present in the trace, sorted.'''
        return sorted({s.kind for s in self.spans})

    def roots(self):
        '''Root spans (no parent), ordered by id.'''
        return sorted((s for s in self.spans if s.parent_id is None),
                      key=lambda s: s.id)

    def subtree(self, span_id):
        '''A span plus all its descendants, ordered by id.'''
        found = [self.span(span_id)]
        for child in self._children(span_id):
            found.extend(self.subtree(child.id))
        return sorted(found, key=lambda s: s.id)

    def max_depth(self):
        '''Longest root-to-leaf chain in the trace.'''
        def depth(span_id):
            kids = self._children(span_id)
            return 1 + max([depth(k.id) for k in kids] or [0])
        return max([depth(r.id) for r in self.roots()] or [0])

    def attribute_keys(self):
        '''Attribute keys used anywhere in the trace, sorted.'''
        keys = set()
        for span in self.spans:
            keys.update(getattr(span, "attributes", {}))
        return sorted(keys)

    def coverage(self, handlers):
        '''Fraction of spans whose kind has a handler.'''
        if not self.spans:
            return 1.0
        handled = sum(1 for s in self.spans if s.kind in handlers)
        return handled / len(self.spans)

    def replay(self, handlers):
        '''handlers: {kind: callable(span, ctx) -> attributes dict}.'''
        self.rng = random.Random(self.seed)
        self.divergences = []
        self.replayed = []
        roots = [s for s in self.spans if s.parent_id is None]
        ctx = {"rng": self.rng, "stubs": {}}
        for root in sorted(roots, key=lambda s: s.id):
            self._replay_span(root, handlers, ctx)
        return {"replayed": len(self.replayed),
                "divergences": list(self.divergences)}

    def _replay_span(self, span, handlers, ctx):
        handler = handlers.get(span.kind)
        produced = handler(span, ctx) if handler else {}
        recorded = dict(getattr(span, "attributes", {}))
        for key, value in produced.items():
            if key in recorded and recorded[key] != value:
                divergence = {"span": span.id, "key": key,
                              "recorded": recorded[key],
                              "replayed": value}
                self.divergences.append(divergence)
                if self.strict:
                    raise Divergence(f"span {span.id}: {key} diverged")
        self.replayed.append(span.id)
        for child in self._children(span.id):
            self._replay_span(child, handlers, ctx)

    def divergence_summary(self):
        '''Aggregate divergences by attribute key and span.'''
        by_key, by_span = {}, {}
        for divergence in self.divergences:
            key = divergence["key"]
            by_key[key] = by_key.get(key, 0) + 1
            sid = divergence["span"]
            by_span[sid] = by_span.get(sid, 0) + 1
        return {"total": len(self.divergences),
                "by_key": by_key, "by_span": by_span}

    def stub_tool(self, name, result):
        '''Register a canned tool result (overrides the recording).'''
        self._stubs = getattr(self, "_stubs", {})
        self._stubs[name] = result
        return self._stubs
