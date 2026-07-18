from parallax.server.engine_core_protocol import EngineCoreFinishReason
from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.request import Request, RequestStatus


class MinimalExecutor(BaseExecutor):
    def handle_input_requests(self, requests):
        raise NotImplementedError

    def process_batch(self, prepared_inputs, return_decoded_tokens=True):
        raise NotImplementedError

    def _prepare_prefill_batch(self, batched_requests):
        raise NotImplementedError

    def _prepare_decode_batch(self, batched_requests):
        raise NotImplementedError

    def _gen_token_id_from_hidden(self, hidden_states):
        raise NotImplementedError

    def check_and_refit_weight(self, refit_weight_path):
        raise NotImplementedError

    def _release_request(self, rid):
        raise NotImplementedError


def test_downstream_batch_failure_is_broadcast_as_terminal_error():
    executor = object.__new__(MinimalExecutor)
    executor.tp_rank = 0
    executor.is_first_peer = False
    executor.is_last_peer = True
    executor.finished_batch = []
    released = []
    executor.release_and_evict_request = released.append
    executor._send_engine_core_terminal_output = lambda **kwargs: None
    request = Request(request_id="failed", routing_table=["head", "tail"])

    executor.fail_batch([request])

    assert released == ["failed"]
    assert executor.finished_batch == [request]
    assert request.status == RequestStatus.ERROR
    assert request.terminal_error is True


def test_first_peer_turns_propagated_pipeline_error_into_frontend_error():
    original = Request(request_id="failed")
    propagated = Request(request_id="failed", status=RequestStatus.ERROR)
    propagated.terminal_error = True

    BaseExecutor.apply_peer_terminal_status(original, propagated)

    assert original.status == RequestStatus.ERROR
    assert original.terminal_error is True
    executor = object.__new__(MinimalExecutor)
    executor.send_to_ipc_socket = None
    assert executor._finish_reason_for_request(original) == (
        EngineCoreFinishReason.ERROR,
        None,
    )
