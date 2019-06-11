import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from peeters_piper.peeter_piper import piper

filename = "GW20130314-0057-s02.csv"

dat = pd.read_csv(filename).iloc[:, :10].values
dat[np.isnan(dat)] = 0

# Plot example data
# Piper plot
fig = plt.figure()
rgb = piper(dat[:, 2:10], "title", use_color=True, fig=fig)
fig.savefig(filename + "_piper_plot.png", dpi=120)
