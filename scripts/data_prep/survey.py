#%%
# Get rough proportion of different datasets
# This will preprocess them according to the selected config - be careful.
from context_general_bci.config.presets import ScaleHistoryDatasetConfig
from context_general_bci.dataset import SpikingDataset

default_cfg = ScaleHistoryDatasetConfig()
chop_assumption = default_cfg.pitt_co.chop_size_ms

dataset_queries = [
    "churchland.*",
    "gallego.*",
    "dyer_co.*",
    "delay.*",
    "miller.*",
    "flint.*",
    "rouse.*",
    "chase.*",
    "mayo.*",
    # "schwartz.*",
    # "hatsopoulos.*",
    "perich.*",
    "pitt_broad.*",
    "chicago.*"
]

timings = {}
for query in dataset_queries:
    default_cfg.datasets = [query]
    dataset = SpikingDataset(default_cfg)
    rough_time_s = len(dataset) * chop_assumption / 1000
    timings[query] = rough_time_s
    print(f"{query}: {rough_time_s:.2f} s")
    print(f"\tMin: {rough_time_s / 60:.2f} min")
    print(f"\tHours: {rough_time_s / 3600:.2f} h")
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# total_h = sum(timings.values()) / 3600
df = pd.DataFrame(timings.items(), columns=["Dataset", "Time (s)"])
df["Time (h)"] = df["Time (s)"] / 3600
# Pie chart
fig, ax = plt.subplots()
ax.pie(timings.values(), labels=timings.keys(), autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f'Data Volume Distribution Total: {df["Time (h)"].sum()}')
plt.show()
#%%
# Show on bar chart
fig, ax = plt.subplots()
sns.barplot(x=list(timings.keys()), y=list(timings.values()), ax=ax)