"""Parallax runtime package defaults."""

import os

# Hugging Face automatically enables hf-xet when the wheel is installed.  The
# Xet client has documented stalls on some home/DNS paths and can wait forever
# inside native threads without honoring the regular HTTP timeout.  Model
# checkpoints are already split into files well below the Hub's HTTP limit, so
# use the resumable HTTP path by default.  Operators can explicitly opt back in
# by exporting HF_HUB_DISABLE_XET=0 before launching the process.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
