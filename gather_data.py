import pandas as pd
import numpy as np
import os

file = open('WISDM_and_shoaib_data/data.txt', 'a')

column_names = ['Time_Stamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Activity_Label']

shoaib_data = pd.read_excel('data.xlsx', names=column_names)


for i in range(len(shoaib_data)):
    file.write('20')
    file.write(',')
    file.write(str(shoaib_data['Activity_Label'][i]))
    file.write(',')
    file.write(str(shoaib_data['Time_Stamp'][i]))
    file.write(',')
    file.write(str(shoaib_data['Ax'][i]))
    file.write(',')
    file.write(str(shoaib_data['Ay'][i]))
    file.write(',')
    file.write(str(shoaib_data['Az'][i]))
    file.write(';')
    file.write('\n')


# 关闭文件
file.close()
