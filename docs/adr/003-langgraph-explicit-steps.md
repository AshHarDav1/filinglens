# ADR-003: LangGraph with explicit steps and limits

Status: accepted (M0, 2026-10-22); to be implemented in M3

## Context

The agent has to choose between three tools (text search, XBRL lookup, calculator), refuse investment
advice, and say "not found in the filings" instead of guessing. A free-form agent loop is hard to bound,
test and debug.

## Decision

Build the agent as a LangGraph graph with named steps: guard, plan, tools (at most 4 rounds), answer,
citation check, plus explicit refuse and not-found paths.

## Consequences

- Every path is visible and testable on its own; the round limit caps cost and latency.
- Less flexible than an open loop; new behaviours need a new node or edge.
- Alternative not picked: a ReAct-style loop with no structure.
