"""
Monkey-patch fix for SentenceTransformer 'Cannot copy out of meta tensor' error.

Volume-mounted into the green agent container. Loaded via PYTHONSTARTUP env var
to patch SentenceTransformer before sqlite_source.py tries to load models.

Root cause: accelerate's device_map="auto" places weights on meta device first,
then dispatches to real devices. On CI runners (7GB RAM, no GPU), this fails.

Fix: Force device="cpu" and remove device_map to load weights directly to CPU.
"""

import os
import sys


def apply_patch():
    """Patch SentenceTransformer to avoid meta tensor errors on CPU-only systems."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        if hasattr(torch, "cuda"):
            torch.cuda.is_available = lambda: False
    except ImportError:
        pass

    try:
        import sentence_transformers
        _original_init = sentence_transformers.SentenceTransformer.__init__

        def _patched_init(self, model_name_or_path=None, *args, **kwargs):
            kwargs["device"] = "cpu"
            kwargs.pop("device_map", None)
            kwargs.setdefault("trust_remote_code", False)

            try:
                _original_init(self, model_name_or_path, *args, **kwargs)
            except NotImplementedError as e:
                if "meta tensor" in str(e):
                    print(f"[fix_st] Meta tensor error, retrying without accelerate: {model_name_or_path}")
                    _retry_without_accelerate(self, model_name_or_path, _original_init, *args, **kwargs)
                else:
                    raise

        def _retry_without_accelerate(self, model_name_or_path, _original_init, *args, **kwargs):
            accelerate_mod = sys.modules.pop("accelerate", None)
            accelerate_utils = sys.modules.pop("accelerate.utils", None)

            try:
                from transformers.utils import import_utils
                orig_check = getattr(import_utils, "is_accelerate_available", None)
                import_utils.is_accelerate_available = lambda: False
            except (ImportError, AttributeError):
                orig_check = None

            try:
                kwargs["device"] = "cpu"
                kwargs.pop("device_map", None)
                _original_init(self, model_name_or_path, *args, **kwargs)
            finally:
                if accelerate_mod is not None:
                    sys.modules["accelerate"] = accelerate_mod
                if accelerate_utils is not None:
                    sys.modules["accelerate.utils"] = accelerate_utils
                if orig_check is not None:
                    try:
                        from transformers.utils import import_utils
                        import_utils.is_accelerate_available = orig_check
                    except (ImportError, AttributeError):
                        pass

        sentence_transformers.SentenceTransformer.__init__ = _patched_init
        print("[fix_st] SentenceTransformer patched for CPU-only loading")

    except ImportError:
        print("[fix_st] sentence_transformers not installed, skip")
    except Exception as e:
        print(f"[fix_st] Patch failed: {e}")


apply_patch()
