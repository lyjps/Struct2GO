from dataclasses import dataclass
import numpy as np
import os

for path,dir_list,file_list in os.walk("/home/jiaops/lyjps/data/struct_feature"):
    for file in file_list:
            data = np.loadtxt((os.path.join(path, file)))
            data = data[data[:,0].argsort()] #按照第0列对行排序
            np.savetxt((os.path.join(path, file)),data, fmt="%d")
