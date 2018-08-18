from metal.modules.base_module import InputModule


class IdentityModule(InputModule):
    """A default identity input module that simply passes the input through."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x
