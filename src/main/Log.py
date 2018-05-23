import data_paths_main as data_paths

def write(text):
    with open(data_paths.log, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text.rstrip('\r\n') + '\n' + content)