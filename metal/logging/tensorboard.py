from tensorboardX import SummaryWriter

from metal.logging.writer import LogWriter


class TensorBoardWriter(LogWriter):
    """Class for logging to Tensorboard during runs, as well as writing simple
    JSON logs at end of runs.

    Stores logs in log_dir/{YYYY}_{MM}_{DD}/{H}_{M}_{S}_run_name.json.
    """

    def __init__(
        self,
        log_dir="tensorboard",
        run_dir=None,
        run_name=None,
        writer_metrics=None,
    ):
        super().__init__(
            log_dir=log_dir,
            run_dir=run_dir,
            run_name=run_name,
            writer_metrics=writer_metrics,
        )

        # Set up TensorBoard summary writer
        self.tb_writer = SummaryWriter(
            self.log_subdir, filename_suffix=f".{run_name}"
        )

    def add_scalar(self, name, val, i):
        if super().add_scalar(name, val, i):
            self.tb_writer.add_scalar(name, val, i)

    def close(self):
        self.write()
        self.tb_writer.close()
