import numpy as np


class OMA:
    """Operational Modal Analysis Class

    This class provides a prototype for all OMA methods
    """

    # Options always stored as a dictionary but specific to the method
    _opts = {}

    def __init__(self, opts: dict) -> None:
        """Initalisation of OMA class

        Basically nothing happens here only managing setting of options

        Args:
            opts (dict): options dictionary
        """
        self.opts = opts

    # Manage user setting options and not overwriting defaults
    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, new_opts):
        # Update non-destructively the options with user provided values
        self._opts.update(new_opts)
