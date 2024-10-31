# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2024/10/31 21:48:59
@Author  :   junewluo 
'''

import datetime  
import inspect  
import os  
import sys  

class Logger:  
    # ANSI color codes  
    RED = '\033[91m'  
    YELLOW = '\033[93m'  # Yellow color
    ENDC = '\033[0m'  # Reset to default color  
  
    def __init__(self, log_file=None, std_out_console=False):  
        self.log_file = log_file
        self.console = std_out_console
        if self.log_file:
            self.file_handler = open(self.log_file, 'a')  
  
    def __del__(self):  
        if hasattr(self, 'file_handler') and self.file_handler:  
            self.file_handler.close()  
  
    def _format_message(self, level, message, frame):  
        now = datetime.datetime.now()  
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')  
        frame_info = inspect.getframeinfo(frame)  
        filename = os.path.basename(frame_info.filename)  
        lineno = frame_info.lineno  
        log_message = f"{timestamp} | {level} | {filename}:{lineno} | {message}"  
        return log_message  
  
    def _print_to_console(self, message, color=None):  
        if color:  
            print(f"{color}{message}{self.ENDC}", file=sys.stdout)  
        else:  
            print(message, file=sys.stdout)  
  
    def info(self, message):  
        log_message = self._format_message('INFO', message, inspect.currentframe().f_back)
        if self.console:  
            self._print_to_console(log_message)  
        if self.log_file:  
            self.file_handler.write(log_message + '\n')  
            self.file_handler.flush()  
  
    def error(self, message):  
        log_message = self._format_message('ERROR', message, inspect.currentframe().f_back)
        if self.console:
            self._print_to_console(log_message, color=self.RED)  
        if self.log_file:  
            self.file_handler.write(log_message + '\n')  
            self.file_handler.flush()
  
    def warning(self, message):  
        log_message = self._format_message('WARNING', message, inspect.currentframe().f_back)
        if self.console:
            self._print_to_console(log_message, color=self.YELLOW)  
        if self.log_file:  
            self.file_handler.write(log_message + '\n')  
            self.file_handler.flush()

# 使用示例
# if __name__ == "__main__":
#     logger = Logger(log_file="example.log", std_out_console=True)
#     logger.info("这是一条普通信息")
#     logger.warning("这是一条警告信息")
#     logger.error("这是一条错误信息")