import datetime
import logging
import os
import os.path as osp

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_logger(logger_name, log_dir=None):

    # Get the current date and time as a string
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Get the name of the script file
    filename = os.path.basename(logger_name)

    # Create the log file name
    log_filename = f'{now}_{filename}.txt'

    # Create a Formatter object
    formatter = logging.Formatter('%(asctime)s %(module)s %(levelname)s: %(message)s')

    # Create a StreamHandler object for stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create a FileHandler object for logging to file
    file_handler = logging.FileHandler(osp.join(log_dir, log_filename), mode='w', delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # file_handler.flush = True

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log some messages
    logger.info('Initialized logger')
    return logger
