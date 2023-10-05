import importlib.util

from peft.import_utils import is_bnb_available

try:
    from peft.import_utils import is_bnb_4bit_available
except ImportError:
    def is_bnb_4bit_available() -> bool:
        if not is_bnb_available():
            return False

        import bitsandbytes as bnb

        return hasattr(bnb.nn, "Linear4bit")


def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def is_flash_attn_available() -> bool:
    import torch.cuda

    return _flash_attn_available and torch.cuda.is_available()


_flash_attn_available = _is_package_available("flash_attn")
