'''Creates module support for the preprocessing-folder to access all files in main and analysis.'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0,parentdir)
import module_support
