import pytest

from parallax.utils.shared_state import SharedState


def test_request_abort_marker_is_explicit_and_clearable():
    state = SharedState({})

    assert state.request_abort_requested("request-1") is False
    state.request_abort("request-1")
    assert state.request_abort_requested("request-1") is True
    state.clear_request_abort("request-1")
    assert state.request_abort_requested("request-1") is False


@pytest.mark.parametrize("request_id", ["", "x" * 257])
def test_request_abort_marker_rejects_invalid_identity(request_id):
    state = SharedState({})

    with pytest.raises(ValueError, match="request_id is invalid"):
        state.request_abort(request_id)
