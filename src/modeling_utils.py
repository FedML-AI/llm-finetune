from typing import Any, Dict, List, Optional, Type, Union

from dataclasses import dataclass, field
import warnings

import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _get_model_class

from .constants import IGNORE_INDEX
from .typing import (
    DataCollatorType,
    is_model_config_type,
    is_model_type,
    ModelConfigType,
    ModelType,
    TokenizerType,
)
from .utils import is_directory


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    escape_token: Optional[str] = field(
        default=None,
        metadata={"help": "If not `None`, will turn off loss for all tokens up until this token"}
    )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.escape_token is not None:
            # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
            # sequence of tokens. This should just be a single token.
            response_token_ids = self.tokenizer.encode(self.escape_token)

            labels = batch["labels"].clone()

            for i in range(len(examples)):
                for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                    response_token_ids_start_idx = idx
                    break
                else:
                    warnings.warn(
                        f"{type(self).__name__} Could not find response key {response_token_ids} in token IDs {batch['labels'][i]}"
                    )

                    response_token_ids_start_idx = len(batch["labels"][i])

                response_token_ids_end_idx = response_token_ids_start_idx + 1

                # Make pytorch loss function ignore all tokens up through the end of the response key
                labels[i, :response_token_ids_end_idx] = IGNORE_INDEX

            batch["labels"] = labels

        return batch


def get_data_collator(
        tokenizer: TokenizerType,
        escape_token: Optional[str] = None,
        pad_to_multiple_of: Optional[int] = 8,
        **kwargs: Any
) -> DataCollatorType:
    _kwargs = dict(
        mlm=False,
        return_tensors="pt"
    )
    _kwargs.update(**kwargs)

    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
        escape_token=escape_token,
        **_kwargs
    )


def get_max_seq_length(model_or_config: Union[str, ModelConfigType, ModelType], **kwargs: Any) -> Optional[int]:
    if is_model_config_type(model_or_config):
        config = model_or_config
    elif is_model_type(model_or_config):
        config = model_or_config.config
    elif isinstance(model_or_config, str):
        config = AutoConfig.from_pretrained(model_or_config, **kwargs)
    else:
        raise TypeError(f"\"{type(model_or_config)}\" is not a supported model_or_config type.")

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        embedding_size = getattr(config, length_setting, None)
        if embedding_size is not None:
            return embedding_size
    else:
        return None


def get_vocab_size(model_or_config: Union[str, ModelConfigType, ModelType], **kwargs: Any) -> Optional[int]:
    if is_model_config_type(model_or_config):
        config = model_or_config
    elif is_model_type(model_or_config):
        config = model_or_config.config
    elif isinstance(model_or_config, str):
        config = AutoConfig.from_pretrained(model_or_config, **kwargs)
    else:
        raise TypeError(f"\"{type(model_or_config)}\" is not a supported model_or_config type.")

    return getattr(config, "vocab_size", None)


# Adapted from https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/auto/auto_factory.py#L407
def get_model_class_from_config(
        config: ModelConfigType,
        cls: _BaseAutoModelClass = AutoModelForCausalLM,
        **kwargs: Any
) -> Type[ModelType]:
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
    has_local_code = type(config) in cls._model_mapping.keys()
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code, config._name_or_path, has_local_code, has_remote_code
    )

    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[cls.__name__]
        if "--" in class_ref:
            repo_id, class_ref = class_ref.split("--")
        else:
            repo_id = config.name_or_path
        model_class: Any = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
        if is_directory(config._name_or_path):
            model_class.register_for_auto_class(cls.__name__)
        else:
            cls.register(config.__class__, model_class, exist_ok=True)
        _ = kwargs.pop("code_revision", None)
        return model_class

    elif type(config) in cls._model_mapping.keys():
        return _get_model_class(config, cls._model_mapping)

    else:
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}."
        )
