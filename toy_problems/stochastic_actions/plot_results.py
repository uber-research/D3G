import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import rc, font_manager
plt.ion()

def getdata(name, num_actions):
    all_results = []

    for i in range(0, 10):
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


for r in [0.25, 0.5, 0.75]:
    plt.clf()

    sns.set(style="ticks", rc={"lines.linewidth": 4})
    plt.figure(figsize=(11,10)) 
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)


    colors = [[1,0,1], [0,1,0], [0,0,1], [1,0,1]]
    sns.set_style("whitegrid")
    c = 0

    for experiment in ["vanilla", "model"]:
        results_1, time = getdata("results/" + str(experiment) + "_stochastic", str(r) + "_4")

        # Plot each line.
        if "vanilla" in experiment:
            name = "QSA"
        else:
            name = "QSS"
        sns.tsplot(
            condition=f"{name}", time=time, data=results_1, color=colors[c], linestyle='-', err_style='ci_bars', interpolate=True, legend=True).set(xlim=(0, 50000), ylim=(-500,0))

        c += 1 


    plt.rc('legend', fontsize=50)    # legend fontsize
    plt.ylabel("Average Episodic Reward",fontsize=30)
    plt.xlabel("Episode",fontsize=30)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)


    plt.show()
    plt.savefig(f"stochastic_{r}.png")
    input("Press enter for the next plot!\n")
