import logging
import os
from datetime import datetime

file_path=os.path.join(os.getcwd(),"logs")
os.makedirs(file_path,exist_ok=True)
file_name=str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.log'
log_file=os.path.join(file_path,file_name)
logging.basicConfig(filename=log_file,level=logging.INFO,
                     format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                     datefmt='%m-%d-%Y %H-%M-%S'
                     )
if __name__=="__main__":
    logging.info("My First Log")
