#Initialization
import numpy as np
import ssapy
from ssapy.utils import Time
import yeager_utils as yu


import matplotlib.pyplot as plt

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# The data we want to access
sim_name = 'data_1_year_1.0GEO_to_18.2GEO'
sim_path = f'/p/lustre2/cislunar/{sim_name}'
fig_path = '/g/g16/yeager7/workdir/cislunar/notebooks/figures/'

def add_to_database(a, b):
    h5_file = 'test.h5'
    key="{a}_{b}"
    data = [a, b]
    yu.append_h5(h5_file, key, data)
    print(yu.read_h5(h5_file, key))

add_to_database(10, 9)