import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use('ggplot')

os.system('grep Test log_test | cut -c16- > performance.csv')
# without header and delimiter value the result can be quite confusing...
a = pd.read_csv('performance.csv',header = None, delimiter=' ')
a.columns = ['h_c', 'accuracy']
plt.plot(a.h_c,a.accuracy)
plt.xlim([0,2])
plt.show()
