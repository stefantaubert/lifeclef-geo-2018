import time
from datetime import datetime

from geo.data_paths import log

def log_start():
    global start_time
    global start_datetime
    start_time = time.time()
    start_datetime = datetime.now()
    print("Start:", start_datetime)

def log_end(modelname, additional=""):
    global start_time
    global start_datetime
    end_date_time = datetime.datetime.now()
    print("End:", end_date_time)
    seconds = time.time() - start_time
    duration_min = round(seconds / 60, 2)
    print("Total duration:", duration_min, "min")
    log_text = str("{}\n--------------------\nStarted: {}\nFinished: {}\nDuration: {}min\n".format
    (
        modelname,
        str(start_datetime),
        str(end_date_time),
        str(duration_min),
    ))
    log_text += additional
    log_text += "============================="
    _write_log(log_text)

def _write_log(text):
    with open(log, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text.rstrip('\r\n') + '\n' + content)
    
    print("#### LOG ####")
    print(text)