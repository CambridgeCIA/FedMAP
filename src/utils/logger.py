def write_to_log(message, log_file="../results/logfile.log"):
    """Write a message to a log file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")