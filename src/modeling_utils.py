from typing import Any, Dict, List, Optional, Type, Union

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


# Adapted from https://github.com/huggingface/trl/blob/01c4a35928f41ba25b1d0032a085519b8065c843/trl/trainer/utils.py#L56
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
            self,
            tokenizer: TokenizerType,
            response_template: str,
            ignore_index: int = IGNORE_INDEX,
            mlm: bool = True,
            mlm_probability: float = 0.15,
            pad_to_multiple_of: Optional[int] = None,
            tf_experimental_compile: bool = False,
            return_tensors: str = "pt"
    ):
        super().__init__(
            tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            tf_experimental_compile=tf_experimental_compile,
            return_tensors=return_tensors
        )

        self.ignore_index = ignore_index

        if len(response_template) == 0:
            raise ValueError(f"{type(self).__name__} requires a non-empty `response_template`.")

        # The prompt ends with the response template. We encode this and then try to find it in the
        # sequence of tokens.
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

        # See https://github.com/huggingface/trl/pull/622
        # See https://github.com/huggingface/trl/issues/598
        # See https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate
        # Some tokenizers such as "GPT2Tokenizer" and "Llama2Tokenizer" tokenize input string differently
        # depending on the context. Below are fallback solutions
        self.response_template_ctx = f"\n{response_template}"
        self.response_ctx_token_ids = self.tokenizer.encode(self.response_template_ctx, add_special_tokens=False)[2:]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                        self.response_token_ids
                        == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx
                    break
            else:
                # Fallback to `response_ctx_token_ids` for tokenizers that requires the input in
                # context (e.g. "GPT2Tokenizer" and "Llama2Tokenizer")
                for idx in np.where(batch["labels"][i] == self.response_ctx_token_ids[0])[0]:
                    if (
                            self.response_ctx_token_ids
                            == batch["labels"][i][idx: idx + len(self.response_ctx_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx
                        break

                else:
                    input_ids = batch['labels'][i][batch['attention_mask'][i] > 0].tolist()

                    warnings.warn(
                        f"{type(self).__name__} Could not find response key `{self.response_template}` in the"
                        f" following instance: ```{self.tokenizer.decode(input_ids)}```"
                        f" This instance will be ignored in loss calculation."
                        f" Note, if this happens often, consider increasing the `max_seq_length`."
                    )

                    # set to the max length of the current sample
                    response_token_ids_start_idx = len(batch["labels"][i])

            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response template
            batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch


def get_data_collator(
        tokenizer: TokenizerType,
        response_template: Optional[str] = None,
        ignore_index: int = IGNORE_INDEX,
        pad_to_multiple_of: Optional[int] = 8,
        mlm: bool = False,
        return_tensors: str = "pt",
        **kwargs: Any
) -> DataCollatorType:
    _kwargs = dict(
        mlm=mlm,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        **kwargs,
    )

    if bool(response_template):
        # if response_template is a non-empty string
        return DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_template,
            ignore_index=ignore_index,
            **_kwargs
        )

    else:
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, **_kwargs)


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
