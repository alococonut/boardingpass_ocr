import datetime
import logging
import os
import sys
import time
import uuid

from asgi_correlation_id.context import correlation_id
from loguru import logger


class Rotator:

    def __init__(self, *, size, at):
        now = datetime.datetime.now()

        self._size_limit = size
        self._time_limit = now.replace(hour=at.hour, minute=at.minute, second=at.second)

        if now >= self._time_limit:
            # The current time is already past the target time so it would rotate already.
            # Add one day to prevent an immediate rotation.
            self._time_limit += datetime.timedelta(days=1)

    def should_rotate(self, message, file):
        file.seek(0, 2)
        if file.tell() + len(message) > self._size_limit:
            return True
        if message.record["time"].timestamp() > self._time_limit.timestamp():
            self._time_limit += datetime.timedelta(days=1)
            return True
        return False


# Rotate file if over 100 MB or at midnight every day
rotator = Rotator(size=1e+8, at=datetime.time(0, 0, 0))

# 日志的路径
log_path = os.path.join(os.getcwd(), 'logs', 'app')
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 日志输出的文件格式
log_path_all = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_log.log')
log_path_info = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_info.log')
log_path_error = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_error.log')

formatter = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | " \
            "{process} | {thread} | {extra[x_request_id]} | " \
            "<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | " \
            "msg: <level>{message}</level>"
logger.remove()
logger.add(sys.stdout, format=formatter)

logger.add(log_path_all, format=formatter, backtrace=True, rotation=rotator.should_rotate,
           retention="1 days", compression='tar.gz',
           enqueue=True, encoding='utf8')
logger.add(log_path_info, format=formatter, backtrace=True, rotation=rotator.should_rotate,
           retention="1 days", compression='tar.gz',
           enqueue=True, encoding='utf8', level='INFO')
# logger.add(log_path_error, format=formatter, backtrace=True, rotation=rotator.should_rotate,
#            retention="1 days", compression='tar.gz',
#            enqueue=True, encoding='utf8', level='ERROR')
logger = logger.patch(lambda record: record["extra"].update(x_request_id=correlation_id.get() or uuid.uuid4().hex))


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def init_uvicorn_logger():
    handler = InterceptHandler()

    logger_name_list = [name for name in logging.root.manager.loggerDict]
    for logger_name in logger_name_list:
        if 'tensorflow' == logger_name:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).setLevel(logging.INFO)
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).addHandler(handler)
        logging.getLogger(logger_name).propagate = False
