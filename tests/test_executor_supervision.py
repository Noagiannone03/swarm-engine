from parallax.server.executor_supervision import (
    ExecutorSupervisionOutcome,
    failed_executor_outcome,
)


def test_memory_contract_failure_requests_a_generation_reload():
    assert (
        failed_executor_outcome(has_memory_contract_failure=True)
        is ExecutorSupervisionOutcome.RELOAD_REQUESTED
    )


def test_unclassified_executor_failure_remains_terminal():
    assert failed_executor_outcome(has_memory_contract_failure=False) is None
