import logging

logging_format = '%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s - %(message)s'
datetime_format = '%Y-%m-%d:%H:%M:%S'

def configure_logging(
        l_format: str = logging_format,
        dt_format: str = datetime_format) -> None:
    """
    Configures logging with specified logging format and datetime.
    :param logging_format: logging messages format
    :param datetime_format: datetime format
    :return: None
    """
    logging.basicConfig(
        format=l_format,
        datefmt=dt_format,
        level=logging.INFO,
    )

def get_logger():
    logger = logging.getLogger(__name__)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(logging_format, datetime_format))
        logger.addHandler(sh)
    return logger