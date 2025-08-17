import matplotlib.pyplot as plt
import numpy as np
import os
import re

result_folders = [
    "burgers_9_results",
    "burgers_11_results",
    "burgers_13_results",
#    "burgers_15_results",
]

loss_histories = dict.fromkeys(result_folders)

for folder in result_folders:
    # read training loss history
    with open(os.path.join(folder, "loss_history.txt")) as f:
        loss_history = [x.strip() for x in f.readlines()]
    loss_histories[folder] = {
        "mean": [float(x.split(",")[0]) for x in loss_history[1:]],
        "std": [float(x.split(",")[1]) for x in loss_history[1:]],
    }


x = [int(re.findall("_([0-9]+)_", key)[0]) for key in result_folders]

# plot last epoch loss vs num qubits
last_epoch_loss = [loss_histories[f"burgers_{num}_results"]["mean"][-1] for num in x]
plt.figure()
plt.plot(x, last_epoch_loss, marker="o")
plt.xlabel("Number of Grid Points")
plt.ylabel("Loss")
plt.title("Last Recorded Training Loss vs Number of Grid Points")
plt.savefig("lastloss_v_gridpoints.png")
plt.close()
