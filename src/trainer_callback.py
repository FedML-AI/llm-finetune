from typing import Any, Dict, Mapping, Optional

from dataclasses import fields, is_dataclass

from torch import Tensor
from torch.nn import Module
from transformers import TrainerState, TrainingArguments
from transformers.integrations import WandbCallback


# Adapted from `transformers.training_args.TrainingArguments.to_dict`
def dataclass_to_dict(dataclass_obj) -> Dict[str, Any]:
    d = {f.name: getattr(dataclass_obj, f.name) for f in fields(dataclass_obj) if f.init}

    for k, v in d.items():
        if k.endswith("_token"):
            d[k] = f"<{k.upper()}>"

    return d


# Adapted from `transformers.training_args.TrainingArguments.to_sanitized_dict`
def dataclass_to_sanitized_dict(dataclass_obj) -> Dict[str, Any]:
    d = dataclass_to_dict(dataclass_obj)

    valid_types = [bool, int, float, str, Tensor]
    return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class HFWandbCallback(WandbCallback):
    def __init__(
            self,
            additional_config: Optional[Mapping[str, Any]] = None,
            *dataclass_objs: Any,
            **dataclass_obj_dict: Any
    ):
        self._wandb = None  # set `_wandb` default value

        super().__init__()

        # convert/shallow copy the additional_config
        additional_config = dict(additional_config) if additional_config is not None else dict()

        # add additional keys
        for dataclass_obj in dataclass_objs:
            if dataclass_obj is None:
                continue

            if not is_dataclass(dataclass_obj):
                raise TypeError(f"all input objects must be dataclasses but got {type(dataclass_obj)}")

            additional_config.update(dataclass_to_sanitized_dict(dataclass_obj))

        for key, dataclass_obj in dataclass_obj_dict.items():
            if dataclass_obj is None:
                continue

            if not is_dataclass(dataclass_obj):
                raise TypeError(f"all input objects must be dataclasses but got {type(dataclass_obj)}")

            additional_config.update({key: dataclass_to_sanitized_dict(dataclass_obj)})

        self.additional_config = additional_config

    def setup(self, args: TrainingArguments, state: TrainerState, model: Module, **kwargs):
        super().setup(args, state, model, **kwargs)

        if self._wandb is not None and state.is_world_process_zero:
            self._wandb.config.update(self.additional_config, allow_val_change=True)
