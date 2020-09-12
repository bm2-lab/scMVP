# -*- coding: utf-8 -*-


__author__ = "Gaoyang Li"
__email__ = "lgyzngc@tongji.edu.cn"
__version__ = "0.0.1"

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
from logging import NullHandler

from ._settings import set_verbosity

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

# default to INFO level logging for the scMVP package
set_verbosity(logging.INFO)

__all__ = ["set_verbosity"]
