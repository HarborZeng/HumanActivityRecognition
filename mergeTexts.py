# -*- coding:utf-8 -*-
# file: FileMerage.py
#

# os模块中包含很多操作文件和目录的函数
import os

# 获取目标文件夹的路径
meragefiledir = os.getcwd() + '\\uciData'
# 获取当前文件夹中的文件名称列表
filenames = os.listdir(meragefiledir)
# 打开当前目录下的result.txt文件，如果没有则创建
# 文件也可以是其他类型的格式，如result.js
file = open('healthData.log', 'w')
# 向文件中写入字符
# file.write('python\n')

# 先遍历文件名
for filename in filenames:
    filepath = meragefiledir + '\\' + filename
    # 遍历单个文件，读取行数
    for line in open(filepath):
        file.writelines(line)
    file.write('\n')

# 关闭文件
file.close()