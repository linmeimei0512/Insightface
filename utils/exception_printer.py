import traceback
import sys


# ================================================
# Exception printer
#
# :param error_message
# ================================================
def exception_printer(error_message):
    # exc_type, exc_value, exc_traceback_obj = sys.exc_info()
    # traceback.print_tb(exc_traceback_obj)
    # traceback.print_exception(exc_type, exc_value, exc_traceback_obj, limit=2, file=sys.stdout)
    # traceback.print_exc(limit=2)
    exception_message = traceback.format_exc(limit=2)

    print('\n' + str(error_message))
    print(exception_message)