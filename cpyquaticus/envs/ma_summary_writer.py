from torch.utils.tensorboard import SummaryWriter

class MultiAgentSummaryWriter:
    def __init__(self, log_dir=None, agents=None):
        self.writer = SummaryWriter(log_dir)
        self.agents = agents if agents else []

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, agent_id=None):
        if agent_id is not None:
            tag = f"agent_{agent_id}/{tag}"
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None, agent_id=None):
        if agent_id is not None:
            main_tag = f"agent_{agent_id}/{main_tag}"
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
    def add_text(self, tag, text_string, global_step=None, agent_id=None):
        """Add a text entry to the summary."""
        if agent_id is not None:
            tag = f"agent_{agent_id}/{tag}"
        self.writer.add_text(tag, text_string, global_step)
    def close(self):
        self.writer.close()