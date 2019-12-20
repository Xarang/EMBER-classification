import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import struct

import sys

N_VECTORS = 10000
VECTOR_SIZE = 2351
FLOAT_SIZE = 4

if len(sys.argv) == 1:
    sys.exit("usage: {} [ xdatfile ] [ ydatfile ]".format(sys.argv[0]))

def get_float_array(byte_array, expected_size):
    #print(len(byte_array))
    data_as_float = struct.unpack("{}f".format(expected_size), byte_array)
    #print(data_as_float)
    #print(len(data_as_float))
    return np.array(data_as_float)

xdatfile = sys.argv[1]
ydatfile = sys.argv[2]
outputfile = "output.png"

xdat = open(xdatfile, 'rb')
xdata = xdat.read(N_VECTORS * VECTOR_SIZE * FLOAT_SIZE)
xdata = get_float_array(xdata, N_VECTORS * VECTOR_SIZE)

ydat = open(ydatfile, 'rb')
ydata = ydat.read(N_VECTORS * FLOAT_SIZE)ss
ydata = get_float_array(ydata, N_VECTORS)


print("[PLOT] opened file and loaded into buffer")

def plot_vector(float_array, color):
    y_range = float_array
    x_range = range(0, VECTOR_SIZE)
    plt.scatter(x_range, y_range, c=[color] * VECTOR_SIZE, cmap='PiYG')

def get_value_color(f):
    if f == 1.0:
        return 1.0
    elif f == 0:
        return 0.0
    elif f == -1:
        return 0.5

colors = list(map(get_value_color, ydata))

vectors = np.split(xdata, len(xdata) / VECTOR_SIZE)

print(len(vectors))
print(len(ydata))

plot_vector(vectors[0], get_value_color(ydata[0]))

#plt.scatter(x_range, y_range, c=colors, cmap='viridis', alpha=0.5)
#plt.plot(sub_arr)

print("[PLOT] plotted file data on graph")

plt.ylabel('value')

plt.ylim((0, 0.1))

plt.savefig(outputfile)

print("[PLOT] saved graph in {}".format(outputfile))