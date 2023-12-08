import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="whitegrid", color_codes=True)

categories = [0.0087, 0.0088, 0.0089, 0.009,  0.0091, 0.0092, 0.0093, 0.0094, 0.0449, 0.045, 0.0452, 0.0453]
core_values = [0.0087, 0.0094, 0.0449, 0.0453]
core_colors = ["#538135", "#89BF65", "#C5E0B3", "#E2EFD9"]

norm=plt.Normalize(min(categories),max(categories))
tuples = list(zip(map(norm,core_values), core_colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

x = np.linspace(0, 6, len(categories))
y = np.zeros((len(categories), ))
colors = cmap(norm(categories))

plt.scatter(x,y,c=categories, cmap=cmap, norm=norm, s=1000)
plt.show()
