import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#hot['HonorSociety'] = np.repeat(1,num_rows)
#hot['LatinHonors'] = np.repeat(1,num_rows)
#hot['ScholarshipsAward'] = np.repeat(1,num_rows)
#hot['CommCollege'] = np.repeat(1,num_rows)
#hot['OtherUni'] = np.repeat(1,num_rows)
#hot['Top100Uni'] = np.repeat(1,num_rows)
#hot['Top10Uni'] = np.repeat(1,num_rows)
#hot['GPAmax'] = np.repeat(4.0,num_rows)

job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
job_dummies_ideal.drop(job_dummies_ideal.columns[1:44], axis=1, inplace=True)

for i in job_dummies_ideal.columns:
  print(i)
  