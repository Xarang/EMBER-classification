import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd

values = []

with open(sys.argv[1], 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        values.append([ i, 'loss', row[0] ])
        values.append([ i, 'accuracy', row[1] ])
        #values.append([ i, 'eval_loss', row[2] ])
        values.append([ i, 'eval_accuracy', row[3] ])
        i += 1

df = pd.DataFrame(values, columns=['epoch', 'label', 'value'])
df.value = pd.to_numeric(df.value)
df = df.pivot(index='epoch', columns='label', values='value')


plt.xlabel('Epoch (n°)')

df.plot()

plt.savefig(sys.argv[2])

print("[DNN PLOT] plotted dnn results in {}".format(sys.argv[2]))