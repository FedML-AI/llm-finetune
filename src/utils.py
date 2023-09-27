from typing import (
    Any,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import inspect
import logging
import os
from pathlib import Path
import shutil

from fedml.arguments import Arguments
import torch.cuda
from torch import distributed as dist, Tensor
from torch.nn import Module
from transformers import HfArgumentParser, Trainer
from transformers.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled
from peft import PeftModel, PromptLearningConfig

from .typing import ModelType, PathType

if is_deepspeed_available():
    import deepspeed

    is_deepspeed_initialized = deepspeed.comm.is_initialized
else:
    def is_deepspeed_initialized() -> bool:
        return False

T = TypeVar("T")


def get_real_path(path: PathType) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def is_file(path: PathType) -> bool:
    return os.path.isfile(get_real_path(path))


def is_directory(path: PathType) -> bool:
    return os.path.isdir(get_real_path(path))


def move_directory_content(src_path: PathType, dest_path: PathType) -> None:
    """
    Move all files/subdirectories in src_path into dest_path then remove src_path.

    Args:
        src_path: source directory path
        dest_path: destination directory path

    Returns:

    """
    if not is_directory(src_path):
        raise FileNotFoundError(f"\"{src_path}\" is not a directory.")
    if is_file(dest_path):
        raise FileExistsError(f"\"{dest_path}\" is an existing file.")

    if get_real_path(src_path) == get_real_path(dest_path):
        return

    src_path = Path(src_path)
    dest_path = Path(dest_path)

    for p in tuple(src_path.iterdir()):
        shutil.move(str(p), str(dest_path / p.relative_to(src_path)))
    shutil.rmtree(str(src_path))


def save_config(model: ModelType, output_dir: PathType) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        """
        adapted from peft.PeftModel.save_pretrained()
        """
        peft_model = model
        model = peft_model.get_base_model()

        for adapter_name, peft_config in peft_model.peft_config.items():
            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    peft_model.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else peft_model.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(str(output_dir))
            peft_config.inference_mode = inference_mode

    model.config.save_pretrained(str(output_dir))


def barrier() -> None:
    if is_deepspeed_initialized():
        deepspeed.comm.barrier()
    elif dist.is_initialized():
        dist.barrier()


def is_deepspeed_module(model: Module) -> bool:
    # TODO: verify
    return any(hasattr(p, "ds_numel") for n, p in model.named_parameters())


def parse_hf_args(
        dataclass_types: Union[Type[T], Iterable[Type[T]]],
        args: Optional[Union[Sequence[str], Arguments, Dict[str, Any]]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> Tuple[T, ...]:
    if parser_kwargs is None:
        parser_kwargs = {}

    parser = HfArgumentParser(dataclass_types, **parser_kwargs)

    if args is None or isinstance(args, Sequence):
        return parser.parse_args_into_dataclasses(args=args, **kwargs)

    elif isinstance(args, Arguments):
        args_dict = dict(args.__dict__)
        if not getattr(args, "using_gpu", True) or torch.cuda.device_count() == 1:
            args_dict.pop("local_rank", None)
            args_dict.pop("device", None)

    elif isinstance(args, dict):
        args_dict = args

    else:
        raise TypeError(f"{type(args)} is not a supported type")

    kwargs.setdefault("allow_extra_keys", True)
    return parser.parse_dict(args_dict, **kwargs)
