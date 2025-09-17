import logging
import traceback

__all__ = ['setup_custom_logger', 'log_exception', 'ColoredLogFormatter'] + logging.__all__

def log_exception(logger, err, fmt="%s"):
    tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
    tb_lines = [ln for lns in tb_lines for ln in lns.splitlines()]
    for tb_line in tb_lines:
        logger.error(fmt, tb_line)


def setup_custom_logger(name: str = "laminography", lfname: str = None, stream_to_console: bool = True, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler
    if lfname:
        f_handler = logging.FileHandler(lfname)
        f_format = logging.Formatter('%(asctime)s - %(name)s(%(lineno)s) - %(levelname)s: %(message)s')
        f_handler.setFormatter(f_format)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)

    # Console handler
    if stream_to_console:
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(ColoredLogFormatter('%(asctime)s - %(message)s'))
        c_handler.setLevel(level)
        logger.addHandler(c_handler)

    return logger


class ColoredLogFormatter(logging.Formatter):
    __BLUE = '\033[94m'
    __GREEN = '\033[92m'
    __RED = '\033[91m'
    __RED_BG = '\033[41m'
    __YELLOW = '\033[33m'
    __ENDC = '\033[0m'

    def _format_message_level(self, message, level):
        colors = {
            'info': self.__GREEN,
            'warning': self.__YELLOW,
            'error': self.__RED,
            'critical': self.__RED_BG,
        }
        if level.lower() in colors:
            message = f"{colors[level.lower()]}{message}{self.__ENDC}"
        return message

    def formatMessage(self, record):
        record.message = self._format_message_level(record.getMessage(), record.levelname)
        return super().formatMessage(record)
