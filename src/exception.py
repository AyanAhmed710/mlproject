import sys
import logging

# Setup logging to a file
logging.basicConfig(
    filename="app.log", 
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s - %(message)s",
    level=logging.INFO
)

def error_message_detail(error, error_detail: sys):
    _, _, exc_info = error_detail.exc_info()
    filename = exc_info.tb_frame.f_code.co_filename

    error_message = "Error occured in python file [{0}] on line number [{1}] with error message [{2}]".format(
        filename, exc_info.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    logging.info("Logging has started")

    try:
        a = 1 / 0   # This will cause ZeroDivisionError
    except Exception as e:
        logging.error("Exception occurred")
        raise CustomException(str(e), sys)
