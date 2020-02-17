from typing import *
import numpy as np
import json
import os
import xlrd
import logging
import xlsxwriter
import datetime
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def sent_email(path: str, key: str):
    def _get_reult():
        results = []
        if os.path.exists(path):
            with open(path, encoding='utf8') as f:
                for line in f.readlines():
                    if line.startswith('INFO') and \
                            line.find(key) != -1:
                        results.append(line)
            return '\n'.join(results)
        else:
            return False

    info = _get_reult()
    if not info: return
    time = datetime.datetime.now().strftime('%y%m%d %H:%M')
    my_sender = os.environ.get('email')  # 发件人邮箱账号
    my_pass = os.environ.get('email_passwd')  # 发件人邮箱密码
    to_user = my_sender  # 收件人邮箱账号，我这边发送给自己
    try:
        msg = MIMEText(info, 'plain', 'utf-8')
        msg['From'] = formataddr(["训练结果", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["我自己", to_user])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = time + ", 训练结果"  # 邮件的主题，也可以说是标题

        server = smtplib.SMTP_SSL("smtp.163.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, [to_user, ],
                        msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception as e:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        raise e
    # print('发送成功')
    return


def label_to_array(labels: List[str], all_labels: List[str]) -> np.ndarray:
    pass


def save_to_json(name: str, obj: Any, dir: str):
    file_name = name + '.json'
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, file_name)
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj=obj, fp=f, ensure_ascii=False, indent=1)


def save_to_txt(name: str, text: str, dir: str):
    file_name = name + '.txt'
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, file_name)
    with open(path, 'w', encoding='utf8') as f:
        f.write(text)


def save_to_xlsx(name, values, header=None):
    wb = xlsxwriter.Workbook(name)
    st = wb.add_worksheet()
    start_row = 0
    if header:
        st.write_row(0, 0, header)
        start_row += 1
    for row in values:
        st.write_row(start_row, 0, row)
        start_row += 1

    wb.close()


def read_xlsx(file_path: str, header: bool = True, index: int = 0):
    sheet = xlrd.open_workbook(filename=file_path).sheet_by_index(index)
    start = int(header)
    for row_index in range(start, sheet.nrows):
        yield sheet.row_values(rowx=row_index, start_colx=0)


def read_csv(file_path: str, header: bool = True, filter: str = ','):
    with open(file_path, 'r', encoding='utf8') as f:
        for row_index, line in enumerate(f.readlines()):
            if row_index == 0 and header: continue
            row_values = line.strip().split(filter)
            yield row_values


def read_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        obj = json.load(f)
    return obj


def real_list_len(l: list) -> int:
    count = 0
    for item in l:
        if item != '':
            count += 1
    return count


def get_abs_path(dir, fname):
    return os.path.join(dir, fname)


def cat_string(strs: List[str]):
    return ''.join(strs)


def generate_time_file(path, suffix):
    now_time = datetime.datetime.now()
    now_time = now_time.strftime('%m%d_%H_%M')
    file = str(now_time) + '.' + suffix
    file_path = os.path.join(path, file)
    return file_path


def re_mkdir(path, exist_ok=True):
    os.makedirs(path, exist_ok=exist_ok)


def save_predict_result(path, data, label, pred):
    assert len(pred) == len(label) == len(data)
    values = []
    for d, t, p in zip(data, label, pred):
        values.append([d, t, p])
    save_to_xlsx(path, values)


def get_logger(path, logger_name):
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                filename=path,
                mode='w',
                encoding='utf8'),
        ],
        level=logging.DEBUG
    )
    logger = logging.getLogger(logger_name)
    return logger


def get_now_time():
    return datetime.datetime.now().strftime('%y%m%d %H:%M:%S')


def get_train_logger(log_path, model_name, data_name):
    path = os.path.join(log_path, model_name, data_name)
    if not os.path.exists(path):
        os.makedirs(path)
    f = generate_time_file(path, 'log')
    return get_logger(f, model_name)


def get_train_checkpoints(checkpoints_path, model_name, data_name):
    path = os.path.join(checkpoints_path, model_name + '_' + data_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return
