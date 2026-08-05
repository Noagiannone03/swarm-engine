# Fabi agentic workload calibration

This directory calibrates placement demand without running a real model. It
uses the official `llm-d-inference-sim` image and its per-token latency, queue,
Prometheus and context-admission behavior. It is an offline qualification tool;
it is not linked into a worker and never receives prompts, accounts or source
code.

The profile is fitted to the qualified Qwen3-4B Mac mini + RTX route. The
observed OpenCode baseline was 12,220 prompt tokens, 4,096 tokens reserved for
output and 16.824 seconds TTFT. With the committed deterministic profile, the
same prefill produced 16.842 seconds on 5 August 2026. The 130 ms ITL represents
roughly 7.7 output tokens/s seen on the lab route. Recalibrate these values when
the model contract, quantization, worker mix or transport changes.

Run the official image pinned by both version and digest:

```console
docker run --rm --network host \
  -v "$PWD/tools/agentic_workload_sim:/profiles:ro" \
  ghcr.io/llm-d/llm-d-inference-sim:v0.10.2@sha256:7f3a1f72875c5dd5d00299ad358dc1f6e17041a5124609a43aa17ad318bbed32 \
  --config /profiles/llm-d-qwen3-4b-lab.yaml
```

Then run the three probes from another shell:

```console
python tools/agentic_workload_sim/probe_llmd_profile.py single
python tools/agentic_workload_sim/probe_llmd_profile.py saturation
python tools/agentic_workload_sim/probe_llmd_profile.py overflow
```

`single` simulates seven output tokens but records the real 4,096-token Fabi
reservation separately. `llm-d-inference-sim` uses `max_tokens` as both the
admission reservation and, with `ignore_eos`, the generated length; generating
all 4,096 tokens at the lab ITL would make every smoke run last about nine
minutes. Fabi's exact 12,220 + 4,096 admission contract remains covered in the
route planner and live E2E tests.

The upstream v0.10.2 running-request metric is updated asynchronously. During
qualification, a first request briefly supplied zero to a load-factor formula
whose valid domain starts at one, reducing a configured factor below 1. The
profile therefore fixes `time-factor-under-load: 1.0`: it uses the simulator to
measure real queue waves, not to invent an unqualified slowdown multiplier.
The measured eight-request probe reached four running and four waiting; the
first wave completed around 2.63 seconds and the second around 5.25 seconds.

The VPS calibration container and its 35 MB image were removed after the run.
No production scheduler container or port was changed.
