import logging

def setup_logging(level=logging.INFO):
    """Configure global logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,   # overwrite any existing handlers (important in notebooks/pytest)
    )