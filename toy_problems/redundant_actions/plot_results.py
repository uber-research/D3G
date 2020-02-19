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

            if t % 1000 == 0:
                results.append(float(reward))
                time.append(float(step))
            t += 1

        all_results.append(results)

    return all_results, time


# Load the data.
sns.set(rc={'figure.figsize':(10,9)})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
colors = [[1,0,0], [0,0,1], [0,1,0], [1,0,1]]
c = 0
for experiment in ["vanilla"]:
    for num_actions in [4, 40, 400]:
        results_1, time = getdata("results/" + str(experiment), num_actions)

        sns.set_style("whitegrid")
        # Plot each line.
        sns.tsplot(
            condition=str(num_actions), time=time, data=results_1, color=colors[c], linestyle='-', err_style='ci_bars', interpolate=True, legend=True).set(xlim=(0, 50000))

        c += 1 


plt.rc('legend', fontsize=20)    # legend fontsize
plt.ylabel("Average Episodic Reward",fontsize=30)
plt.xlabel("Episode",fontsize=30)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=0.)

plt.show()
