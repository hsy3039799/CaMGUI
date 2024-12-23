import os
import logging

def setup_logger(args):
    # 创建 /logs 文件夹如果它不存在
    logs_dir = os.path.join(os.getcwd(), 'logs')
    
    # 根据输入参数构造日志文件名
    log_file_name = args.log_name
    log_file_path = os.path.join(logs_dir, log_file_name)
    
    # 配置日志
    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # 确保之前的handler不被重复添加
    if not logger.handlers:
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger