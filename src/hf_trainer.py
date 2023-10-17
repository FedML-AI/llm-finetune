from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import math
from pathlib import Path
import shutil

from torch import Tensor
from torch.nn import Module
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .configurations import DatasetArguments, ModelArguments
from .distributed import barrier
from .integrations import is_fedml_available
from .trainer_callback import FedMLCallback, HFWandbCallback
from .typing import (
    DataCollatorType,
    DatasetType,
    LrSchedulerType,
    ModelType,
    OptimizerType,
    PathType,
    TokenizerType,
)
from .utils import move_directory_content, is_directory


class HFTrainer(Trainer):
    def __init__(
            self,
            model: Union[ModelType, Module] = None,
            args: TrainingArguments = None,
            model_args: ModelArguments = None,
            dataset_args: DatasetArguments = None,
            data_collator: Optional[DataCollatorType] = None,
            train_dataset: Optional[DatasetType] = None,
            eval_dataset: Optional[Union[DatasetType, Dict[str, DatasetType]]] = None,
            tokenizer: Optional[TokenizerType] = None,
            model_init: Optional[Callable[[], Union[ModelType, Module]]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[OptimizerType, LrSchedulerType] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.model_args = model_args
        self.dataset_args = dataset_args

        if is_fedml_available():
            self.add_callback(FedMLCallback)

        # replace WandbCallback if exist
        self.replace_callback(
            old_callback=WandbCallback,
            new_callback=HFWandbCallback(model_args=self.model_args, dataset_args=self.dataset_args)
        )

    def log(self, logs: Dict[str, float]) -> None:
        # Adapted from https://github.com/huggingface/transformers/blob/b71f20a7c9f3716d30f6738501559acf863e2c5c/examples/pytorch/language-modeling/run_clm.py#L630-L634
        # compute perplexity
        for key in tuple(logs.keys()):
            if key.endswith("loss"):
                prefix = key[:key.rfind("loss")]

                try:
                    perplexity = math.exp(logs[key])
                except OverflowError:
                    perplexity = math.inf
                logs[f"{prefix}perplexity"] = perplexity

        super().log(logs)

    def replace_callback(
            self,
            old_callback: Union[Type[TrainerCallback], TrainerCallback],
            new_callback: Union[Type[TrainerCallback], TrainerCallback],
    ) -> None:
        # replace callback if exists
        if self.pop_callback(old_callback) is not None:
            self.add_callback(new_callback)

    def save_checkpoint(self, output_dir: PathType) -> None:
        output_dir = Path(output_dir)
        checkpoint_dir = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # should not save a temporary checkpoint if the checkpoint folder already exist
        should_save_temp_ckpt = not is_directory(checkpoint_dir)
        barrier()

        model_wrapped = self.model_wrapped if self.model_wrapped is not None else self.model

        if should_save_temp_ckpt:
            self._save_checkpoint(model_wrapped, trial=None)

        with self.args.main_process_first(local=self.args.save_on_each_node):
            if self.args.should_save:
                output_dir.mkdir(parents=True, exist_ok=True)
                if should_save_temp_ckpt:
                    # if saved a temporary checkpoint, should move all its content into target directory
                    move_directory_content(checkpoint_dir, output_dir)
                else:
                    # TODO: dirs_exist_ok is for python 3.8+, support older python
                    shutil.copytree(str(checkpoint_dir), str(output_dir), dirs_exist_ok=True)
