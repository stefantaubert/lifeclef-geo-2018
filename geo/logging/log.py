from geo.data_paths import log

def write(text):
    with open(log, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text.rstrip('\r\n') + '\n' + content)