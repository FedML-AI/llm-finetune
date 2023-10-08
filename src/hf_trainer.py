from typing import Dict

import math
from pathlib import Path
import shutil

from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .distributed import barrier
from .typing import PathType
from .utils import move_directory_content, is_directory


class HFTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        # compute perplexity
        # Adapted from https://github.com/huggingface/transformers/blob/b71f20a7c9f3716d30f6738501559acf863e2c5c/examples/pytorch/language-modeling/run_clm.py#L630-L634
        for key in tuple(logs.keys()):
            if key.endswith("loss"):
                prefix = key[:key.rfind("loss")]

                try:
                    perplexity = math.exp(logs[key])
                except OverflowError:
                    perplexity = float("inf")
                logs[f"{prefix}perplexity"] = perplexity

        super().log(logs)

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
