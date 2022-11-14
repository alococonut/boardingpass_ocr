import os

# 日志的路径
log_path = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 指定端口
bind = "0.0.0.0:6000"
# 并行工作进程数
workers = 4
# 进程并发线程数
threads = 2
# 监听队列:未决连接的最大数量，即等待服务的客户的数量。默认2048个，一般不修改
backlog = 2048
# 工作模式协程
worker_class = "uvicorn.workers.UvicornWorker"
# 设置守护进程
daemon = False
# 设置最大并发量（每个worker处理请求的工作线程数，正整数，默认为1）
worker_connections = 2000
# 最大客户端并发数量，默认情况下这个值为1000。
max_requests = 2000
# pid 写入文件
pidfile = f'{log_path}/gunicorn.pid'
# 设置访问日志路径
accesslog = f'{log_path}/gunicorn_access.log'
# 设置gunicorn访问日志格式，错误日志无法设置(实测：access_log_format不生效)
access_log_format = '%({x_request_id}o)s %(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
# 设置错误信息日志路径
errorlog = f'{log_path}/gunicorn_server.log'
# 设置错误信息日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
loglevel = 'info'

# 进程名
proc_name = 'gunicorn_process'
# 设置超时时间120s，默认为30s。按自己的需求进行设置timeout = 120
timeout = 120
# 超时重启
graceful_timeout = 300
# 在keep-alive连接上等待请求的秒数，默认情况下值为2。一般设定在1~5秒之间。
keepalive = 3
# HTTP请求行的最大大小，此参数用于限制HTTP请求行的允许大小，默认情况下，这个值为4094。
# 值是0~8190的数字。此参数可以防止任何DDOS攻击
limit_request_line = 5120
# 限制HTTP请求中请求头字段的数量。
# 此字段用于限制请求头字段的数量以防止DDOS攻击，与limit-request-field-size一起使用可以提高安全性。
# 默认情况下，这个值为100，这个值不能超过32768
limit_request_fields = 101
# 限制HTTP请求中请求头的大小，默认情况下这个值为8190。
# 值是一个整数或者0，当该值为0时，表示将对请求头大小不做限制
limit_request_field_size = 8190
# Load application code before the worker processes are forked.
# By preloading an application you can save some RAM resources as well as speed up server boot times. Although,
# if you defer application loading to each worker process, you can reload your application code easily by restarting
# workers.
preload_app = True
