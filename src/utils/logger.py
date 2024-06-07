RESULT_LOG_FILE = "./results/logfile.log"

def log_results(message):
    """Write a message to a log file."""
    with open(RESULT_LOG_FILE, "a") as f:
        f.write(message + "\n")