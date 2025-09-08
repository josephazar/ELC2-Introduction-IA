import seaborn as sns
import numpy as np
import pandas as pd
tips = pd.read_csv("data/tips.csv")

size = tips["size"]
size.loc[:15] = np.nan
size.head(20)

size.shape

size.isnull().sum()

mean = size.mean()
mean = round(mean)
print(mean)

size.fillna(mean, inplace=True)
size.head(20)

"""_Outliers_"""

import matplotlib.pyplot as plt
plt.hist(size)
plt.show()

min_val = size.mean() - (3 * size.std())
min_val

max_val = size.mean() + (3 * size.std())
max_val

outliers = size[size > max_val]
outliers.count()

outliers

size = size[size <= max_val]
size.shape

"""### Exercise 1.03
__Applying Feature Engineering over Text Data__
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd

enc = LabelEncoder()
tips["sex"] = enc.fit_transform(tips['sex'].astype('str'))
tips["smoker"] = enc.fit_transform(tips['smoker'].astype('str'))
tips["day"] = enc.fit_transform(tips['day'].astype('str'))
tips["time"] = enc.fit_transform(tips['time'].astype('str'))

tips.head()

"""### Exercise 1.04
__Normalizing and Standardizing Sata__

_Normalization_
"""

tips_normalized = (tips - tips.min())/(tips.max()-tips.min())
tips_normalized.head(10)

"""_Standardization_"""

tips_standardized = (tips - tips.mean())/tips.std()
tips_standardized.head(10)