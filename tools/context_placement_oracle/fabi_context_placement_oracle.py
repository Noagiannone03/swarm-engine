"""Standalone CP-SAT oracle for small Fabi placement scenarios."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Demand:
    context_tokens: int
    target_slots: int
    weight: int


@dataclass(frozen=True)
class Option:
    worker_id: str
    option_id: str
    start_layer: int
    end_layer: int
    context_tokens: int
    max_sessions: int


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_scenario(payload: dict[str, Any]) -> tuple[int, tuple[Demand, ...], tuple[Option, ...]]:
    if set(payload) != {"num_layers", "context_classes", "demands", "workers"}:
        raise ValueError("scenario contains missing or unknown top-level fields")
    num_layers = _positive_int(payload["num_layers"], "num_layers")
    context_classes = tuple(
        _positive_int(value, "context class") for value in payload["context_classes"]
    )
    if context_classes != tuple(sorted(set(context_classes))):
        raise ValueError("context classes must be strictly increasing and unique")

    demands = tuple(
        Demand(
            context_tokens=_positive_int(item["context_tokens"], "demand context"),
            target_slots=_positive_int(item["target_slots"], "target_slots"),
            weight=_positive_int(item["weight"], "demand weight"),
        )
        for item in payload["demands"]
    )
    demand_contexts = tuple(item.context_tokens for item in demands)
    if demand_contexts != context_classes:
        raise ValueError("demands must exactly match the ordered context classes")

    options: list[Option] = []
    worker_ids: set[str] = set()
    for worker in payload["workers"]:
        if set(worker) != {"worker_id", "options"}:
            raise ValueError("worker contains missing or unknown fields")
        worker_id = worker["worker_id"]
        if not isinstance(worker_id, str) or not worker_id or worker_id in worker_ids:
            raise ValueError("worker ids must be non-empty and unique")
        worker_ids.add(worker_id)
        option_ids: set[str] = set()
        for item in worker["options"]:
            if set(item) != {
                "option_id",
                "start_layer",
                "end_layer",
                "context_tokens",
                "max_sessions",
            }:
                raise ValueError("placement option contains missing or unknown fields")
            option_id = item["option_id"]
            if not isinstance(option_id, str) or not option_id or option_id in option_ids:
                raise ValueError("option ids must be non-empty and unique per worker")
            option_ids.add(option_id)
            start = item["start_layer"]
            end = item["end_layer"]
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= num_layers
            ):
                raise ValueError("placement option has an invalid layer span")
            context_tokens = _positive_int(item["context_tokens"], "option context")
            if context_tokens not in context_classes:
                raise ValueError("placement option uses an unsigned context class")
            options.append(
                Option(
                    worker_id=worker_id,
                    option_id=option_id,
                    start_layer=start,
                    end_layer=end,
                    context_tokens=context_tokens,
                    max_sessions=_positive_int(item["max_sessions"], "max_sessions"),
                )
            )
    return num_layers, demands, tuple(options)


def solve(payload: dict[str, Any], *, max_time_seconds: float = 30.0) -> dict[str, Any]:
    """Return an exact small-scenario optimum with shared cross-class slots."""

    if max_time_seconds <= 0:
        raise ValueError("solver time limit must be positive")
    num_layers, demands, options = _parse_scenario(payload)
    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(f"selected_{option.worker_id}_{option.option_id}")
        for index, option in enumerate(options)
    }
    by_worker: dict[str, list[int]] = {}
    for index, option in enumerate(options):
        by_worker.setdefault(option.worker_id, []).append(index)
    for indices in by_worker.values():
        model.add(sum(selected[index] for index in indices) <= 1)

    flows: dict[tuple[int, int], cp_model.IntVar] = {}
    for demand_index, demand in enumerate(demands):
        for option_index, option in enumerate(options):
            if option.context_tokens < demand.context_tokens:
                continue
            flow = model.new_int_var(
                0,
                option.max_sessions,
                f"flow_c{demand.context_tokens}_o{option_index}",
            )
            model.add(flow <= option.max_sessions * selected[option_index])
            flows[demand_index, option_index] = flow

    # A selected placement has one shared KV pool. A long-context option can
    # serve a shorter class, but the same session slot cannot serve both at the
    # same instant.
    for option_index, option in enumerate(options):
        option_flows = [
            flow
            for (demand_index, index), flow in flows.items()
            if index == option_index
        ]
        if option_flows:
            model.add(sum(option_flows) <= option.max_sessions * selected[option_index])

    served: dict[int, cp_model.IntVar] = {}
    for demand_index, demand in enumerate(demands):
        served[demand_index] = model.new_int_var(
            0, demand.target_slots, f"served_c{demand.context_tokens}"
        )
        for boundary in range(num_layers + 1):
            outgoing = [
                flow
                for (class_index, option_index), flow in flows.items()
                if class_index == demand_index and options[option_index].start_layer == boundary
            ]
            incoming = [
                flow
                for (class_index, option_index), flow in flows.items()
                if class_index == demand_index and options[option_index].end_layer == boundary
            ]
            if boundary == 0:
                model.add(sum(outgoing) - sum(incoming) == served[demand_index])
            elif boundary == num_layers:
                model.add(sum(incoming) - sum(outgoing) == served[demand_index])
            else:
                model.add(sum(incoming) == sum(outgoing))

    selection_penalty_bound = len(options) + 1
    model.maximize(
        sum(
            demand.weight * selection_penalty_bound * served[index]
            for index, demand in enumerate(demands)
        )
        - sum(selected.values())
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise RuntimeError(f"placement oracle did not find a solution: {solver.status_name(status)}")

    chosen = [
        {
            "worker_id": option.worker_id,
            "option_id": option.option_id,
            "start_layer": option.start_layer,
            "end_layer": option.end_layer,
            "context_tokens": option.context_tokens,
            "max_sessions": option.max_sessions,
        }
        for index, option in enumerate(options)
        if solver.value(selected[index])
    ]
    return {
        "status": solver.status_name(status).lower(),
        "placements": chosen,
        "served_slots_by_context": {
            str(demand.context_tokens): solver.value(served[index])
            for index, demand in enumerate(demands)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--max-time-seconds", type=float, default=30.0)
    args = parser.parse_args()
    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    print(json.dumps(solve(payload, max_time_seconds=args.max_time_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
