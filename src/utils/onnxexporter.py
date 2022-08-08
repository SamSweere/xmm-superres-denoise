import os
import pytorch_lightning as pl
import torch


class OnnxExporter(pl.Callback):
    def __init__(self, in_dims, wandb_logger, save_on_start=True, save_on_end=True):
        super().__init__()

        self.in_dims = in_dims
        self.wandb_logger = wandb_logger
        self.save_on_start = save_on_start
        self.save_on_end = save_on_end
        self.export_path = wandb_logger.experiment.dir

        self.onnx_path = os.path.join(self.export_path, "onnx")
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)

    def save_onnx(self, model_filename, pl_module):
        dummy_input = torch.zeros(self.in_dims, device=pl_module.device)

        model_path = os.path.join(self.onnx_path, model_filename)

        # TODO: check if this actually fixes the issue
        input_names = ["input"]
        if hasattr(pl_module, "heats"):
            input_names.append("heat")
            dummy_input = (
                dummy_input,
                torch.tensor(pl_module.heats[0]).type_as(dummy_input),
            )

            # TODO: the inverse operation in permutations.py makes exporting to onnx impossible, might be fixable with custom algorithm

        if not hasattr(pl_module, "heats"):
            torch.onnx.export(pl_module, dummy_input, model_path, opset_version=11)
            # Export the model
            torch.onnx.export(
                pl_module,  # model being run
                dummy_input,  # model input (or a tuple for multiple inputs)
                model_path,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=11,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=input_names,  # the model's input names
                output_names=["output"],  # the model's output names
                dynamic_axes={
                    "input": {0: "batch_size"},  # variable length axes
                    "output": {0: "batch_size"},
                },
            )
            self.wandb_logger.experiment.save(model_path, base_path=self.export_path)

    def on_sanity_check_end(self, trainer, pl_module):
        # Save the model at the start
        if trainer.fast_dev_run or not self.save_on_start:
            # Do not try to save in fast dev run mode
            return

        # Save the model to wandb
        model_filename = f"model_{str(trainer.current_epoch).zfill(5)}.onnx"
        self.save_onnx(model_filename, pl_module)

    # def on_validation_epoch_end(self, trainer, pl_module):

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.fast_dev_run or not self.save_on_end:
            # Do not try to save in fast dev run mode
            return

        model_filename = "model_final.onnx"
        self.save_onnx(model_filename, pl_module)
