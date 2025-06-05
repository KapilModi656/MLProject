import logging
import os
from datetime import datetime
logger=logging.getlogger(__name__)
file_path=os.path.join(os.getcwd(),"logs")
def main():
    file_name=str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.log'
    logging.basic_config(filename=file_name,level=logging.info)
    