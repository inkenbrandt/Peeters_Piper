import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from peeter_piper import piper

filename = "GW20130314-0057-s02_additional_field.csv"

df = pd.read_csv(filename)

# Plot example data
# Piper plot
fig = plt.figure()
markers = ["s", "o", "^", "v", "+", "x"]
arrays = []
for i, (label, group_df) in enumerate(df.groupby("additional-field")):
    arr = group_df.iloc[:, 2:10].values
    arrays.append(
        [
            arr,
            {
                "label": label,
                "marker": markers[i],
                "edgecolor": "k",
                "linewidth": 0.3,
                "facecolor": "none",
            },
        ]
    )

rgb = piper(arrays, "title", use_color=True, fig=fig)
plt.legend()
fig.savefig(filename + "_piper_plot.png", dpi=120)
