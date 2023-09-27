import math
from pathlib import Path

from peft import PeftModel
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME as HF_WEIGHTS_NAME

from .utils import is_file


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if state.is_world_process_zero or (state.is_local_process_zero and args.save_on_each_node):
            # see https://github.com/huggingface/peft/issues/96#issuecomment-1460080427
            checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            model = kwargs.get("model", None)

            # TODO: support shard loading; see transformers.modeling_utils.load_sharded_checkpoint
            checkpoint_path = checkpoint_dir / HF_WEIGHTS_NAME
            adapter_checkpoint_path = checkpoint_dir / PEFT_WEIGHTS_NAME
            if isinstance(model, PeftModel) and not is_file(adapter_checkpoint_path):
                # backward compatibility
                assert is_file(checkpoint_path)

                # when using DeepSpeed Zero 3, model weights need to be converted.
                # conversion is done by Trainer, we need to load the saved weights manually
                checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

                model.save_pretrained(str(checkpoint_dir), state_dict=checkpoint)

        return control


class PauseResumeCallback(TrainerCallback):
    def __init__(
            self,
            start_global_step: int = -1,
            start_epoch: float = -1,
            step_threshold: float = math.inf,
            epoch_threshold: float = math.inf
    ):
        if (start_epoch < 0) != (start_global_step < 0):
            raise ValueError(
                f"start_epoch and start_global_step must both be negative or both be non-negative,"
                f" but received start_epoch = {start_epoch}, start_global_step = {start_global_step}."
            )

        self.start_global_step = start_global_step
        self.start_epoch = start_epoch
        self.step_threshold = step_threshold
        self.epoch_threshold = epoch_threshold

    @property
    def use_step_threshold(self) -> bool:
        return not math.isinf(self.step_threshold)

    @property
    def use_epoch_threshold(self) -> bool:
        return not self.use_step_threshold and not math.isinf(self.epoch_threshold)

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if self.start_global_step < 0 and self.start_epoch < 0:
            # if both values are unset
            self.start_epoch = state.epoch
            self.start_global_step = state.global_step
        else:
            # recover from previous run
            state.epoch = self.start_epoch
            state.global_step = self.start_global_step

        return control

    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if state.global_step - self.start_global_step >= self.step_threshold:
            control.should_training_stop = True

        elif self.use_epoch_threshold and state.epoch - self.start_epoch >= self.epoch_threshold:
            # epoch is a float; partial epoch is allowed which means it needs to be checked every step
            control.should_training_stop = True

        return control

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if state.epoch - self.start_epoch >= self.epoch_threshold:
            control.should_training_stop = True

        return control

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if args.max_steps is not None and state.global_step < args.max_steps:
            control.should_training_stop = False

        elif args.max_steps is None and args.num_train_epochs is not None and state.epoch < args.num_train_epochs:
            control.should_training_stop = False

        # save training progress for resuming
        self.start_global_step = state.global_step
        self.start_epoch = state.epoch

        return control
