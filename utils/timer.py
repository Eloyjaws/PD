import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

logger.setLevel("DEBUG")
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)


start_time = None

def start_timer():
    global start_time
    start_time = time.time()

def end_timer_and_print(local_msg, fn_name=""):
    end_time = time.time()
    run_time = end_time - start_time
    print(local_msg)
    logger.debug(f"{fn_name} ran in {run_time:.3f} secs")