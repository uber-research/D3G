import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import rc, font_manager



def getdata(name, num_actions):
    all_results = []

    for i in range(0, 50):
        f = open(name + "_" + str(num_actions) + "_" +  str(i) + ".txt", "r")
        results = []
        time = []
      
        t = 0
        for line in f:
            step, reward = (line.replace("\n", "")).split(" ")

            if t % 10 == 0:
                results.append(float(reward))
                time.append(float(step))
        
            if t >= 10000:
                break 

            t += 1

        all_results.append(results)

    return all_results, time

# Load the data.

sns.set(rc={'figure.figsize':(10,9)})
sns.set(style="ticks", rc={"lines.linewidth": 2})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
colors = [[1,0,0], [0,0,1], [0,1,0], [1,0,1]]

exp = "vanilla"
c = 0
for experiment in ["vanilla_shuffle", "model_shuffle", "vanilla_original_shuffled", "model_original_shuffled"]:
    for num_actions in [4]:
        results_1, time = getdata("results/" + str(experiment), num_actions)

        sns.set_style("whitegrid")
        color = colors[c]

        if "vanilla_shuffle" in experiment:
            name = "QSA transfer"
        elif "model_shuffle" in experiment:
            name = "QSS transfer"
        elif "vanilla_original" in experiment:
            name = "QSA scratch"
        else:
            name = "QSS scratch"

        # Plot each line.
        sns.tsplot(
            condition=str(name), time=time, data=results_1, color=color, linestyle='-', err_style='ci_bars', ci=95, interpolate=True, legend=True).set(xlim=(0, 10000))

        c += 1 


plt.rc('legend', fontsize=30)    # legend fontsize
plt.ylabel("Average Episodic Reward",fontsize=20)
plt.xlabel("Episode",fontsize=20)
plt.legend(loc='upper right')
plt.savefig(exp + "_actions.png")

plt.show()
