import json
from types import SimpleNamespace

from parallax_utils.fabi_events import emit
from parallax.server.executor import factory


def test_emit_writes_one_stable_machine_readable_line(capsys):
    emit(
        "allocated",
        start_layer=15,
        end_layer=28,
        context_tokens=32768,
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("[FABI] ")
    payload = json.loads(line.removeprefix("[FABI] "))
    assert payload["event"] == "allocated"
    assert payload["start_layer"] == 15
    assert payload["end_layer"] == 28
    assert payload["context_tokens"] == 32768
    assert isinstance(payload["ts"], float)


def test_emit_coerces_non_json_fields_without_raising(capsys):
    emit("peer_id", peer_id={1, 2})

    payload = json.loads(capsys.readouterr().out.strip().removeprefix("[FABI] "))
    assert payload["event"] == "peer_id"
    assert payload["peer_id"] == "{1, 2}"


def test_executor_process_emits_loading_before_materialization(monkeypatch):
    events = []

    class Executor:
        def run_loop(self):
            events.append(("run_loop", {}))

        def shutdown(self):
            events.append(("shutdown", {}))

    monkeypatch.setattr(factory, "set_log_level", lambda _level: None)
    monkeypatch.setattr(factory, "create_from_args", lambda *_args: Executor())
    monkeypatch.setattr(
        factory,
        "emit_fabi_event",
        lambda event, **fields: events.append((event, fields)),
    )
    args = SimpleNamespace(
        log_level="INFO",
        tp_rank=0,
        start_layer=15,
        end_layer=28,
        planned_context_tokens=32768,
        allocation_epoch=4,
    )

    factory.run_executor_process(args)

    assert events == [
        (
            "weights_load_start",
            {
                "start_layer": 15,
                "end_layer": 28,
                "context_tokens": 32768,
                "allocation_epoch": 4,
            },
        ),
        ("run_loop", {}),
        ("shutdown", {}),
    ]
