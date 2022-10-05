#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import paramiko
import scp
import os
import time

# 远程目录
remote_path = "~/image_segmentation"

# 服务器ip地址
host = "10.134.2.18"

# 端口
port = 22

# 用户名
username = "zhuhaoze"

# 密码
password = "haoze332"

# 需要传输的文件或者文件夹
local_paths = ["main.py", "mask.py", "utils.py", "img.png", "datasets.py"]

# 连接
print("Connecting...")
sshclient = paramiko.SSHClient()
sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy)
sshclient.connect(host, username = username, password=password, port=port)
scpclient = scp.SCPClient(sshclient.get_transport(), socket_timeout=15.0)

# 传输文件
print("Connected!")
print("Start to update")
for files in local_paths :
    if os.path.exists(files) :
        print("Updating: " + files)
        scpclient.put(files, remote_path=os.path.join(remote_path, files + "_temp"), recursive=os.path.isdir(files))
    else :
        print("Cannot find the file or directory: " + files)
        print("Update interrupt")
        for fil in local_paths:
            if fil == files:
                break
            sshclient.exec_command("rm" + (" -r " if os.path.isdir(fil) else ' ') + os.path.join(remote_path, fil + "_temp"))
        exit()

for files in local_paths :
    sshclient.exec_command("rm" + (" -r " if os.path.isdir(files) else ' ') + os.path.join(remote_path, files))
    sshclient.exec_command("mv " + os.path.join(remote_path, files + "_temp") + " " + os.path.join(remote_path, files))
print("Update successfully!")

# 重启网站
# print("Restarting the website...")
# sshclient.exec_command("cd " + remote_path + ';' + "nohup python3 webstart.py > nohup_log/nohup_web.log 2>&1 &")
# print("Complete!")

# 断开连接
sshclient.close()
print("Exit")
