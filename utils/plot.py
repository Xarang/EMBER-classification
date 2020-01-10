import numpy as np
import matplotlib.pyplot as plt

"""
-- mean repartition score: 0.14954200330684916
-- mean cluster 0 score: 0.01740078400388967
-- mean cluster 1 score: 0.11087445845535737
-- mean computation time: 88.71811056137085
ref
"""

mean_repartition_scores = [0.846904418113224, 0.8314861843980669, 0.8184361055038134, 0.7880872255709344]
mean_cluster_scores = [
    (0.8046706690115468 + 0.7957337975608527) / 2,
    (0.7888400638410424 + 0.8056201801703486) / 2,
    (0.7861452138989583 + 0.7703210312072301) / 2,
    (0.7878007244657117 + 0.7911883959498865) / 2
]

mean_computation_time = [
    4.389638786315918, #128
    2.445586745738983, #64
    1.9646186780929566,#48
    1.546538031101227  #32
]

indexes = [128, 64, 48, 32]

fig, ax1 = plt.subplots()

ax1.set_xlabel('Feature Vector Size')
ax1.set_ylabel('seconds')
line, = ax1.plot(indexes, mean_computation_time, marker='o', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
#ax2.set_ylabel('Mean Repartition Score')
line1, = ax2.plot(indexes, mean_repartition_scores, marker='o', color='tab:blue')
line1.set_label('Mean Repartition Score')
line2, = ax2.plot(indexes, mean_cluster_scores, marker='o', color='tab:green')
line2.set_label('Mean Cluster Scores')

ax2.legend()
plt.savefig('out.png')