import numpy as np
import pandas as pd
from datetime import datetime
import os

filename = os.path.join('Data', 'WARRIGAL_RD N of HIGH STREET_RD.xlsx')
data = pd.read_excel(filename)
window_size = 5
x, y = [], []
for i in range(0, len(data)):
    x.append(data.iloc[i, 0:window_size].tolist())
    y.append(data.iloc[i, 5])

x = np.array(x)
y = np.array(y)