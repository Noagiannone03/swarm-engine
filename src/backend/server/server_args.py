import argparse
import os

from parallax_utils.logging_config import get_logger
from parallax.p2p.liveness import (
    DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS,
    validate_scheduler_heartbeat_timeout,
)

logger = get_logger(__name__)


def _heartbeat_timeout_default() -> str:
    # Keep this as a string so argparse applies ``type`` to an environment
    # default and reports a normal CLI usage error instead of a traceback.
    return os.environ.get(
        "PARALLAX_HEARTBEAT_TIMEOUT",
        str(DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS),
    )


def _heartbeat_timeout_arg(raw: str) -> float:
    try:
        return validate_scheduler_heartbeat_timeout(float(raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Lattica configuration
    parser.add_argument("--initial-peers", nargs="+", default=[], help="List of initial DHT peers")
    parser.add_argument("--relay-servers", nargs="+", default=[], help="List of relay DHT peers")
    parser.add_argument(
        "--announce-maddrs", nargs="+", default=[], help="List of multiaddresses to announce"
    )
    parser.add_argument("--tcp-port", type=int, default=0, help="Port for Lattica TCP listening")
    parser.add_argument("--udp-port", type=int, default=0, help="Port for Lattica UDP listening")
    parser.add_argument("--dht-prefix", type=str, default="gradient", help="Prefix for DHT keys")

    # Scheduler configuration
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=3001, help="Port to listen on")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    parser.add_argument("--init-nodes-num", type=int, default=None, help="Number of initial nodes")
    parser.add_argument(
        "--allocation-strategy",
        choices=["dp", "greedy"],
        default="dp",
        help="Layer allocation strategy",
    )
    parser.add_argument(
        "--routing-strategy",
        choices=["dp", "rr"],
        default="dp",
        help="Request routing strategy; dp dynamically allocates newly joined workers",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=_heartbeat_timeout_arg,
        default=_heartbeat_timeout_default(),
        help=(
            "Worker lease TTL in seconds. It must exceed one heartbeat RPC deadline "
            "and its retry interval."
        ),
    )
    parser.add_argument(
        "--is-local-network", type=bool, default=True, help="Whether to use local network"
    )
    parser.add_argument(
        "--use-hfcache",
        action="store_true",
        default=False,
        help="Use local Hugging Face cache only (no network download)",
    )

    # Weight refit configuration
    parser.add_argument(
        "--enable-weight-refit", type=bool, default=False, help="Enable online weight refit"
    )
    parser.add_argument(
        "--weight-refit-mode",
        type=str,
        default="disk",
        help="Refit mode to choose where. Choices 'cpu' or 'disk'",
    )

    args = parser.parse_args()

    return args
