import pandas as pd
import numpy as np

def l2(a,b):
    su=0
    i=0
    df1=pd.read_excel('/home/sunny/Desktop/VAFL/Client_Selection/karmany_/Client/karmany/karmany_P.xlsx')
    df2=pd.read_excel('/home/sunny/Desktop/VAFL/Client_Selection/karmany_/Client/karmany/karmany_Pn.xlsx')
    df1=df1.values.astype(np.float32)
    df2=df2.values.astype(np.float32)

    for x in range(df1.shape[-1]):
        print(x)
        for y in range(df2.shape[-1]):
            i+=1
            su+=(sum((df1[:,x]-df2[:,y])**2))**0.5
    return su/i
print("L2:",l2(1,2))
