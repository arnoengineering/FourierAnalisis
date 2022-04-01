import numpy as np
import os
from svgutils import transform as sg

# from matplotlib import pyplot as plt
file = '../Extra/Do_Mayor_armadura.svg'
poly_file = '../media/poly_file.csv'
n_file = '../Extra/scale.svg'

path = os.path.dirname(__file__)
max_size = 10
os.chdir(path)

# get path from svg,
fig = sg.fromfile(file)
or_size = fig.get_size()
or_size = np.array([float(o) for o in or_size])

size_ratio = max_size / max(or_size)
size = or_size * 2
fig_el = fig.getroot()
fig.set_size([str(o) for o in size])


# for path ie line el:

fig.save(n_file)
print(fig.to_str())
