#!/usr/bin/python3
# coding=utf-8
import os
from pathlib import Path
import logging
from logging.config import dictConfig

# Constant directory & files
DIR_BASE = Path(os.path.dirname(os.path.realpath(__file__))).parent
DIR_RESOURCES = DIR_BASE / 'resources'
DIR_MODELS = DIR_RESOURCES / 'models'

RAW_MODEL_NAME = 'model.h5'
RAW_METAS_NAME = 'meta.json'
ZIP_MODEL_PREFIX = 'model.zip'


dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "format"  : "[%(asctime)s] [%(process)d:%(thread)d] [%(levelname)s] [%(name)s] %(filename)s:%(funcName)s:%(lineno)d %(message)s",
            "datefmt" : "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "basic",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {
        "pkmodels" : {
            "handlers" : ["console"],
            "propagate": "true",
            "level"    : "INFO"
        }
    }
})

logger = logging.getLogger(__name__)
logger.info('logs will be printed in console')