import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    return "error occurred in script [{0}] at lineno [{1}] with message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)
    def __str__(self):
        return self.error_message
    def __repr__(self):
        return CustomException.__name__.str()+" "+self.error_message