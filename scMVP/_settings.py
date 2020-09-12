import logging
from typing import Union

logger = logging.getLogger(__name__)
scMVP_logger = logging.getLogger("scMVP")

autotune_formatter = logging.Formatter(
    "[%(asctime)s - %(processName)s - %(threadName)s] %(levelname)s - %(name)s\n%(message)s"
)


class DispatchingFormatter(logging.Formatter):
    """Dispatch formatter for logger and it's sub logger."""

    def __init__(self, default_formatter, formatters=None):
        super().__init__()
        self._formatters = formatters if formatters is not None else {}
        self._default_formatter = default_formatter

    def format(self, record):
        # Search from record's logger up to it's parents:
        logger = logging.getLogger(record.name)
        while logger:
            # Check if suitable formatter for current logger exists:
            if logger.name in self._formatters:
                formatter = self._formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # If no formatter found, just use default:
            formatter = self._default_formatter
        return formatter.format(record)


def set_verbosity(level: Union[str, int]):
    """Sets logging configuration for scMVP based on chosen level of verbosity.

    Sets "scMVP" logging level to `level`
    If "scMVP" logger has no StreamHandler, add one.
    Else, set its level to `level`.
    """
    scMVP_logger.setLevel(level)
    has_streamhandler = False
    for handler in scMVP_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)
            logger.info(
                "'scMVP' logger already has a StreamHandler, set its level to {}.".format(
                    level
                )
            )
            has_streamhandler = True


    if not has_streamhandler:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s | %(message)s"
        )
        ch.setFormatter(
            DispatchingFormatter(formatter, {"scMVP.autotune": autotune_formatter})
        )
        scMVP_logger.addHandler(ch)
        logger.info("Added StreamHandler with custom formatter to 'scMVP' logger.")
