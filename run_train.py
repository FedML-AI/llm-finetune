from typing import Any, Optional, Tuple, Union

from datetime import timedelta
import logging
from pathlib import Path
from timeit import default_timer as timer

from accelerate.utils import compare_versions
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.configurations import DatasetArguments, ModelArguments
from src.constants import DEFAULT_MAX_SEQ_LENGTH
from src.dataset_utils import get_keyword_replacer, get_prompt_formatter
from src.hf_trainer import HFTrainer
from src.integrations import is_deepspeed_zero3_enabled
from src.modeling_utils import (
    get_data_collator,
    get_max_seq_length as _get_max_seq_length,
    get_model_class_from_config,
    get_vocab_size,
)
from src.models import add_flash_attention
from src.typing import DatasetType, ModelConfigType, ModelType, TokenizerType
from src.utils import parse_hf_args, save_config


def preprocess_dataset(
        dataset_args: DatasetArguments,
        dataset: DatasetType,
        tokenizer: TokenizerType
) -> DatasetType:
    if {"input", "output"}.issubset(dataset.column_names):
        # This is required for medical meadow
        dataset = dataset.rename_columns({
            "input": "context",
            "output": "response",
        })

    remove_columns = list({"text", *dataset.column_names})
    if "text" not in dataset.column_names:
        dataset = dataset.map(get_prompt_formatter(dataset_args.prompt_style))
    dataset = dataset.map(get_keyword_replacer())

    tokenization_kwargs = dict(
        truncation=dataset_args.truncate_long_seq,
        max_length=dataset_args.truncation_max_length,
    )

    logging.info(f"preprocessing dataset")
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], **tokenization_kwargs),
        batched=True,
        remove_columns=remove_columns
    )

    if dataset_args.remove_long_seq and dataset_args.max_seq_length is not None:
        dataset = dataset.filter(lambda rec: len(rec["input_ids"]) <= dataset_args.max_seq_length)

        if hasattr(dataset, "num_rows"):
            logging.info(f"dataset has {dataset.num_rows:,} rows after filtering for truncated records")

    if hasattr(dataset, "num_rows"):
        logging.info(f"dataset has {dataset.num_rows:,} rows")

    return dataset


def get_dataset(
        dataset_args: DatasetArguments,
        tokenizer: TokenizerType,
        seed: Optional[int] = None
) -> Tuple[DatasetType, DatasetType]:
    dataset_kwargs = dict(
        path="json",
        name=dataset_args.dataset_config_name,
        streaming=dataset_args.dataset_streaming,
    )

    if dataset_args.dataset_name is not None:
        dataset_kwargs["path"] = dataset_args.dataset_name
        dataset_kwargs["data_files"] = None

    elif len(dataset_args.dataset_path) >= 2:
        dataset_kwargs["data_files"] = {"train": dataset_args.dataset_path[0], "test": dataset_args.dataset_path[1]}

    elif len(dataset_args.dataset_path) == 0:
        raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

    else:
        dataset_kwargs["data_files"] = dataset_args.dataset_path

    dataset_dict = load_dataset(**dataset_kwargs)
    if dataset_args.cleanup_data_cache:
        dataset_dict.cleanup_cache_files()
    if len(dataset_dict.keys()) == 1:
        dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)

        logging.info("splitting dataset")
        dataset_dict = dataset.train_test_split(
            test_size=dataset_args.test_dataset_size,
            shuffle=True,
            seed=seed
        )

        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
    else:
        train_dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)
        test_dataset = preprocess_dataset(dataset_args, dataset_dict["test"], tokenizer)

    logging.info(f"done preprocessing")

    if hasattr(train_dataset, "num_rows"):
        logging.info(f"Train data size: {train_dataset.num_rows:,}")
    if hasattr(test_dataset, "num_rows"):
        logging.info(f"Test data size: {test_dataset.num_rows:,}")
    return train_dataset, test_dataset


def get_tokenizer(model_args: ModelArguments, **kwargs) -> TokenizerType:
    kwargs.setdefault("trust_remote_code", True)

    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_args: ModelArguments, tokenizer_length: Optional[int] = None, **kwargs) -> ModelType:
    kwargs.setdefault("trust_remote_code", True)
    torch_dtype = kwargs.pop("torch_dtype", model_args.torch_dtype)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    model_cls = get_model_class_from_config(config, **kwargs)

    use_transformers_flash_attn = False
    if (
            compare_versions("transformers", ">=", "4.34.0")
            and getattr(model_cls, "_supports_flash_attn_2", False)
    ):
        # starting from transformers v4.34.0, some HF models support flash attention v2
        # only enable `use_flash_attention_2` flag for supported models; other models will be
        # patched by `add_flash_attention`
        # see https://github.com/huggingface/transformers/issues/26350
        kwargs.setdefault("use_flash_attention_2", model_args.use_flash_attention)
        use_transformers_flash_attn = model_args.use_flash_attention

    if model_args.load_pretrained:
        kwargs.setdefault("low_cpu_mem_usage", not is_deepspeed_zero3_enabled())

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs
        )
    else:
        if use_transformers_flash_attn:
            # As of transformers v4.34.0, `AutoModel.from_config` does not support `use_flash_attention_2` flag
            config = model_cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=kwargs.get("device_map", None)
            )

        # see https://discuss.huggingface.co/t/how-to-load-model-without-pretrained-weight/34155/3
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

    model_vocab_size = get_vocab_size(model.config)
    if tokenizer_length is not None and model_vocab_size < tokenizer_length:
        logging.info(f"Resize embedding from {model_vocab_size:,} to tokenizer length: {tokenizer_length:,}")
        model.resize_token_embeddings(tokenizer_length)

        # model.config should also be updated
        assert model.config.vocab_size == tokenizer_length

    if model_args.use_flash_attention and not use_transformers_flash_attn:
        # patch models that do not support `use_flash_attention_2`
        add_flash_attention(model)

    if model_args.use_lora:
        if model_args.lora_on_all_modules:
            from src.peft_utils import LORA_LAYER_TYPES

            additional_target_modules = []
            for n, m in model.named_modules():
                if isinstance(m, tuple(LORA_LAYER_TYPES)):
                    additional_target_modules.append(n.split(".")[-1])

            if len(additional_target_modules) > 0:
                if model_args.lora_target_modules is None:
                    model_args.lora_target_modules = []
                model_args.lora_target_modules = list(set(model_args.lora_target_modules + additional_target_modules))

        # apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            target_modules=model_args.lora_target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
        )
        # see https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if model_args.lora_on_all_modules:
            from peft.tuners.lora import LoraLayer

            # enable gradient for non-LoRA layers
            lora_layer_prefixes = tuple({n for n, m in model.named_modules() if isinstance(m, LoraLayer)})

            for n, p in model.named_parameters():
                if not n.startswith(lora_layer_prefixes):
                    p.requires_grad = True

    if torch_dtype is not None:
        logging.info(f"Loading model in {model_args.model_dtype}.")
        model.to(torch_dtype)
        model.config.torch_dtype = torch_dtype

    return model


def get_max_seq_length(
        model_or_config: Union[str, ModelConfigType, ModelType],
        default_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        **kwargs: Any
) -> int:
    embedding_size = _get_max_seq_length(model_or_config, **kwargs)

    if embedding_size is not None:
        logging.info(f"Found max length: {embedding_size}")
    else:
        embedding_size = default_max_seq_length
        logging.info(f"Using default max length: {embedding_size}")

    return embedding_size


def train() -> None:
    # configs
    model_args, dataset_args, training_args = parse_hf_args((ModelArguments, DatasetArguments, TrainingArguments))

    # prepare models
    logging.info(f"Loading tokenizer for \"{model_args.model_name_or_path}\"")
    tokenizer = get_tokenizer(model_args)

    logging.info(f"Loading model for \"{model_args.model_name_or_path}\"")
    model = get_model(model_args, tokenizer_length=len(tokenizer), use_cache=not training_args.gradient_checkpointing)

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)

    # dataset
    with training_args.main_process_first():
        train_dataset, test_dataset = get_dataset(
            dataset_args=dataset_args,
            tokenizer=tokenizer,
            seed=training_args.seed
        )

    trainer = HFTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=get_data_collator(
            tokenizer,
            response_template=dataset_args.response_template,
            # set to `pad_to_multiple_of` to max_seq_length so that all distributed processes share the same
            # sequence length. This is required for computing metrics.
            pad_to_multiple_of=dataset_args.max_seq_length
        )
    )

    # log training time
    start_time = timer()

    if training_args.do_train:
        final_output_dir = Path(training_args.output_dir) / "final"

        if trainer.args.should_save:
            # save model config before training
            save_config(model, final_output_dir)

        logging.info("Training")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        logging.info(f"Saving model to \"{training_args.output_dir}\"")
        trainer.save_checkpoint(final_output_dir)

    # log training time
    end_time = timer()
    logging.info(f"[{training_args.process_index}] total training time: {timedelta(seconds=end_time - start_time)}")

    if training_args.do_eval:
        logging.info("Evaluating")
        logging.info(trainer.evaluate())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train()
