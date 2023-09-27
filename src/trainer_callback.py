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
