import logging
import initialize_logger


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    initialize_logger.initialize_logger(logger)
    logger.info("Hello World!")
