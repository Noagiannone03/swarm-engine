# Fabi context placement oracle

This is an offline CP-SAT reference solver. It is intentionally isolated from
the Parallax worker environment because OR-Tools 9.15 requires protobuf 6.x,
while the qualified engine runtime requires protobuf 7.x. The oracle exchanges
plain JSON with simulation tooling and never becomes a live placement
authority.

Create a dedicated virtual environment, install this directory, then pass an
input JSON file:

```console
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/fabi-context-placement-oracle scenario.json
.venv/bin/python oracle_checks.py
```

Each worker has mutually exclusive placement options. Each option is a
contiguous layer edge with a qualified context class and a session capacity.
For every demand class CP-SAT creates a conserved source-to-tail flow. Flows
share the selected option's capacity across classes, so one long-context slot
cannot be counted again as simultaneous short-context capacity.

The oracle keeps two separate conserved flows: concurrent KV session slots use
each option's `max_sessions`, while independent routes cap every worker option
at one. Both capacities are shared across context classes, and route capacity
has the same two-to-one priority used by the shadow heuristic. This prevents a
single high-concurrency failure domain from being reported as route redundancy.
