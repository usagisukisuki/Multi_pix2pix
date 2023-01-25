import numpy as np 
import pandas as pd
import json

#PATH = "predict.txt"
#with open(PATH, mode = 'w') as f:
#    pass

log1 = json.load(open('result_1/log'))
log2 = json.load(open('result_2/log'))
log3 = json.load(open('result_3/log'))

df_result_1 = pd.DataFrame(log1)
df_result_2 = pd.DataFrame(log2)
df_result_3 = pd.DataFrame(log3)
df1 = df_result_1.drop(['epoch','elapsed_time'], axis=1)
df2 = df_result_2.drop(['epoch','elapsed_time'], axis=1)
df3 = df_result_3.drop(['epoch','elapsed_time'], axis=1)
#print(df1)
df1 = np.array(df1)
df2 = np.array(df2)
df3 = np.array(df3)
#print(df1)
df1 = df1[50:]
df2 = df2[50:]
df3 = df3[50:]
a1 = np.argmax(df1[:,1])
a2 = np.argmax(df2[:,1])
a3 = np.argmax(df3[:,1])
#a = np.argmax(df[:,1])
#iou = df[a]
iou1 = df1[a1]
iou2 = df2[a2]
iou3 = df3[a3]
print(iou1)

iou = (iou1+iou2+iou3)/3.0
#print(iou)
#with open(PATH, mode = 'a') as f:
#    f.write("\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (iou[0], iou[1], iou[2], iou[3], iou[4], iou[5], iou[6], iou[7], iou[8], iou[9]))



