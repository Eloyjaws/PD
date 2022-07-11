import logging
import time
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


start_times = dict({})


def log(message):
    logging.info(message)


def start_timer(event="default"):
    global start_times
    start_time = time.time()
    start_times[event] = start_time


def end_timer_and_print(event="default"):
    end_time = time.time()
    started_at = start_times.get(event)
    del start_times[event]
    run_time = end_time - started_at
    logging.info(f"{event} ran in {run_time:.3f} secs")
