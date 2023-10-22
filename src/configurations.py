from typing import List, Optional, Union

from dataclasses import dataclass, field
import os
import warnings

from accelerate.utils import compare_versions
from datasets import get_dataset_split_names
import torch
from transformers import TrainingArguments

from .constants import (
    CUSTOM_LOGGERS,
    DATASET_NAMES,
    MODEL_DTYPES,
    MODEL_DTYPE_MAPPING,
    MODEL_NAMES,
    PEFT_TYPES,
    PROMPT_STYLES,
)
from .dataset_utils import RESPONSE_KEY, RESPONSE_KEY_NL
from .typing import to_torch_dtype
from .utils import is_directory, is_file


@dataclass
class ExperimentArguments(TrainingArguments):
    custom_logger: List[str] = field(
        default_factory=list,
        metadata={
            "help": "The list of customized logger to report the results and logs to.",
            "choices": CUSTOM_LOGGERS,
            "nargs": "+",
        }
    )

    def __post_init__(self):
        if "none" in self.custom_logger:
            self.custom_logger = []
        elif "all" in self.custom_logger:
            self.custom_logger = [l for l in CUSTOM_LOGGERS if l not in ("all", "none")]

        super().__post_init__()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="EleutherAI/pythia-70m", metadata={"help": "Model name or path."})
    model_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model data type. Set to \"none\" to use the default data type.",
            "choices": MODEL_DTYPES,
        }
    )
    peft_type: str = field(
        default="none",
        metadata={"help": "PEFT type. Set to \"none\" to disable PEFT.", "choices": PEFT_TYPES}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_on_all_modules: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA on all supported layers."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
            "nargs": "+",
        }
    )
    auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Authentication token for Hugging Face private models such as Llama 2."}
    )
    load_pretrained: bool = field(default=True, metadata={"help": "Whether to load pretrained model weights."})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention."})

    def __post_init__(self) -> None:
        if is_file(self.model_name_or_path):
            if self.load_pretrained:
                raise ValueError(
                    "`model_name_or_path` must be a model ID or directory path if `load_pretrained=True`."
                )

        elif not is_directory(self.model_name_or_path):
            # if model_name_or_path is not a local file or directory
            if self.model_name_or_path not in MODEL_NAMES:
                model_names_str = "', '".join(MODEL_NAMES)

                raise ValueError(
                    f"`model_name_or_path` must be a valid file/directory path or a supported model ID"
                    f" (choose from '{model_names_str}') but received \"{self.model_name_or_path}\"."
                )

            if self.model_name_or_path.startswith("meta-llama/Llama-2-"):
                if compare_versions("transformers", "<", "4.31.0"):
                    raise NotImplementedError(f"{self.model_name_or_path} requires transformers >= 4.31.0")

                if self.auth_token is not None:
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = str(self.auth_token)

                # need to verify if already logged in
                from huggingface_hub import HfApi
                from huggingface_hub.utils import LocalTokenNotFoundError

                try:
                    HfApi().whoami()
                except LocalTokenNotFoundError:
                    raise LocalTokenNotFoundError(
                        f"Token is required for {self.model_name_or_path}, but no token found. You need to provide a"
                        f" token or be logged in to Hugging Face."
                        f"\nTo pass a token, you could pass `--auth_token \"<your token>\"` or set environment"
                        f" variable `HUGGING_FACE_HUB_TOKEN=\"${{your_token}}\"`."
                        f"\nTo login, use `huggingface-cli login` or `huggingface_hub.login`."
                        f" See https://huggingface.co/settings/tokens for detail."
                    )

        if self.model_dtype is not None:
            # convert model_dtype to canonical name
            self.model_dtype = MODEL_DTYPE_MAPPING[self.model_dtype]

    @property
    def torch_dtype(self) -> Optional[torch.dtype]:
        return to_torch_dtype(self.model_dtype)


@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Hugging Face dataset name. If set to an non-empty string, will override `dataset_path`.",
            "choices": [""] + DATASET_NAMES,
        }
    )
    dataset_path: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Path to dataset file(s). If contains multiple entries, the 1st entry is considered"
                    " the training split and the 2nd entry is the test split.",
            "nargs": "+",
        }
    )
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "dataset configuration name"})
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the data progressively while iterating on the dataset."}
    )
    test_dataset_size: int = field(
        default=-1,
        metadata={
            "help": "The test dataset size. Will be ignored if set to a non-positive value, if `dataset_name`"
                    " contains \"test\" split, or if `dataset_path` has at least 2 entries.",
        }
    )
    test_dataset_ratio: Optional[float] = field(
        default=-1.0,
        metadata={
            "help": "Test dataset ratio. If set to a valid value (`0 < test_dataset_ratio < 1`) will override"
                    " `test_dataset_size`. Will be ignored if set to an invalid value, if `dataset_name`"
                    " contains \"test\" split, or if `dataset_path` has at least 2 entries.",
        }
    )
    eval_dataset_size: int = field(
        default=-1,
        metadata={
            "help": "The evaluation dataset size. This dataset is used to evaluate the model performance"
                    " during training. Set to a non-positive number to use the test dataset for the"
                    " evaluation during training.",
        }
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The max sequence length. If unspecified, will use the context length of the model."})
    truncate_long_seq: bool = field(
        default=True,
        metadata={"help": "Whether to truncate long sequences whose length > max_seq_length."}
    )
    remove_long_seq: bool = field(
        default=False,
        metadata={"help": "Whether to remove all data whose token length > max_seq_length."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of processes when generating cache."}
    )
    tokenize_on_the_fly: bool = field(
        default=False,
        metadata={"help": "Whether to tokenize the input on-the-fly."}
    )
    prompt_style: str = field(
        default="dolly",
        metadata={"help": "Prompt template style.", "choices": PROMPT_STYLES}
    )
    response_template: str = field(
        default=RESPONSE_KEY_NL,
        metadata={
            "help": f"The response template for instruction fine-tuning such as `{RESPONSE_KEY}`. If set to"
                    f" a non-empty string, The response template and all text before it will not be included"
                    f" in the loss computation.",
        }
    )
    cleanup_data_cache: bool = field(
        default=False,
        metadata={
            "help": f"Whether to cleanup the data cache before data preprocessing. By default the `datasets`"
                    f" library caches preprocessed data on disk. When developing/changing the data preprocessing"
                    f" logic we need to clean the data cache to ensure the most up-to-date data is generated.",
        }
    )

    def __post_init__(self) -> None:
        if not bool(self.dataset_name):
            # if `dataset_name` is None or empty string
            self.dataset_name = None

        if bool(self.dataset_name):
            # if `dataset_name` is a valid string
            self.dataset_path = []

            split_names = get_dataset_split_names(self.dataset_name, self.dataset_config_name)
            if len(split_names) <= 1 and self.test_size is None:
                raise ValueError(
                    f"`{self.dataset_name}` only has 1 split. A positive `test_dataset_ratio`"
                    f" or `test_dataset_size` is required."
                )

        elif len(self.dataset_path) <= 0:
            # if dataset_name is None
            raise ValueError("`dataset_name` and `dataset_path` cannot both be empty.")

        elif len(self.dataset_path) > 2:
            warnings.warn("More than 2 dataset paths provided. Only the first 2 will be loaded.")
            self.dataset_path = self.dataset_path[:2]

        elif len(self.dataset_path) == 1 and self.test_size is None:
            raise ValueError(
                "A positive `test_dataset_ratio` or `test_dataset_size` is required when"
                " `dataset_path` has only 1 entry."
            )

        if self.dataset_streaming:
            if self.tokenize_on_the_fly:
                warnings.warn(
                    "`dataset_streaming` is not compatible with `tokenize_on_the_fly`."
                    " Setting `tokenize_on_the_fly` to \"False\"."
                )
                self.tokenize_on_the_fly = False

            if self.remove_long_seq:
                warnings.warn(
                    "`dataset_streaming` is not compatible with `remove_long_seq`."
                    " Setting `remove_long_seq` to \"False\"."
                )
                self.remove_long_seq = False

            if self.eval_dataset_size > 0:
                warnings.warn(
                    f"Using `dataset_streaming` with `eval_dataset_size={self.eval_dataset_size}`"
                    f" may slow down the data processing since this requires partially downloading the"
                    f" dataset."
                )

        if self.remove_long_seq and self.tokenize_on_the_fly:
            raise ValueError("`remove_long_seq` is not compatible with `tokenize_on_the_fly`")

        if self.remove_long_seq and not self.truncate_long_seq:
            warnings.warn("`truncate_long_seq` is set to \"True\" since `remove_long_seq` is \"True\".")
            self.truncate_long_seq = True

        if self.dataset_num_proc is not None and self.dataset_num_proc <= 0:
            warnings.warn("Received non-positive `dataset_num_proc`; fallback to CPU count.")
            self.dataset_num_proc = os.cpu_count()

    @property
    def test_size(self) -> Optional[Union[int, float]]:
        if 0 < self.test_dataset_ratio < 1:
            return self.test_dataset_ratio

        elif self.test_dataset_size > 0:
            return self.test_dataset_size

        else:
            return None

    @property
    def truncation_max_length(self) -> Optional[int]:
        if self.max_seq_length is not None and self.remove_long_seq:
            # set to max_seq_length + 1 so that sequences with length >= max_seq_lengths can be
            # filtered out by removing all entries with length > max_seq_length.
            return self.max_seq_length + 1
        else:
            return self.max_seq_length
