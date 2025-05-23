# utils/logging_setup.py
import os
import logging

def setup_logging(verbosity: str):
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    # Default to INFO if verbosity is invalid or not found
    log_level = log_levels.get(verbosity.upper(), logging.INFO)

    # Remove existing handlers to avoid duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized with level: {logging.getLevelName(log_level)}")

def get_verbosity(args_verbosity: str) -> str:
    """Gets verbosity level from args or environment variable."""
    env_verbosity = os.getenv("LOG_VERBOSITY")
    # Prioritize command line argument over environment variable
    if args_verbosity:
        return args_verbosity
    elif env_verbosity:
        return env_verbosity
    else:
        return "INFO" # Default verbosity