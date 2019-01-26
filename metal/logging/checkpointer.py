import os
import shutil

import torch


class Checkpointer(object):
    def __init__(
        self, checkpoint_dir="checkpoints", checkpoint_runway=0, verbose=True
    ):
        """Saves checkpoints as applicable based on a reported metric.

        Args:
            checkpoint_runway (int): don't save any checkpoints for the first
                this many iterations
            checkpoint_dir (str): the directory for saving checkpoints
        """
        self.best_model = None
        self.best_iteration = None
        self.best_score = None
        self.checkpoint_runway = checkpoint_runway
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.state = {}

        if checkpoint_runway and verbose:
            print(
                f"No checkpoints will be saved in the first "
                f"checkpoint_runway={checkpoint_runway} iterations."
            )

    def checkpoint(self, model, iteration, score, optimizer, lr_scheduler):
        if iteration >= self.checkpoint_runway:
            self.state["epoch"] = iteration
            self.state["model"] = model.state_dict()
            self.state["optimizer"] = optimizer.state_dict()
            self.state["lr_scheduler"] = (
                lr_scheduler.state_dict() if lr_scheduler else None
            )
            self.state["score"] = score

            is_best = score > self.best_score
            if is_best:
                if self.verbose:
                    print(
                        f"Saving model at iteration {iteration} with best "
                        f"score {score:.3f}"
                    )
                self.best_model = True
                self.best_iteration = iteration
                self.best_score = score
                self.state["best_iteration"] = iteration
                self.state["best_score"] = score

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            torch.save(
                self.state,
                f"{self.checkpoint_dir}/model_checkpoint_{iteration}.pth",
            )

            # Copies the model's best iteration (checkpoint) to a seperate file to reload after training
            if is_best:
                shutil.copyfile(
                    f"{self.checkpoint_dir}/model_checkpoint_{iteration}.pth",
                    f"{self.checkpoint_dir}/best_model.pth",
                )

    def load_best_model(self, model):
        if self.best_model is None:
            raise Exception(
                f"Best model was never found. Best score = "
                f"{self.best_score}"
            )
        if self.verbose:
            print(
                f"Restoring best model from iteration {self.best_iteration} "
                f"with score {self.best_score:.3f}"
            )
            state = torch.load(
                f"{self.checkpoint_dir}/best_model.pth",
                map_location=torch.device("cpu"),
            )
            self.best_iteration = state["epoch"]
            self.best_score = state["score"]
            model.load_state_dict(state["model"])
            return model

    def restore(self, destination):
        state = torch.load(f"{destination}")
        return state
