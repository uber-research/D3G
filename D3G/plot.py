import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.ion() 

envs = ['Reacher-v2','Reacher-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Ant-v2', 'Humanoid-v2']

policies = ['TD3', 'OurDDPG', 'D3G',  'Standard_QSS']

colors = [[1,0,1], [0,0,1], [0,1,0], 'y']

seeds = range(10)

for env in envs:
  c = 0 
  plt.clf() 
  plt.figure(figsize=(10,9)) 
  sns.set(style="ticks", rc={"lines.linewidth": 2})
  matplotlib.rc('xtick', labelsize=20)
  matplotlib.rc('ytick', labelsize=20)

  sns.set_style("whitegrid")
  for policy in policies:
    base_file_name = policy + "_" + env
    try:
      if "Standard" in policy:
        plot_name = "D3G (no cycle)"
      elif "DDPG" in policy:
        plot_name = "DDPG"
      else:
        plot_name = policy

      results = [np.load("results/" + base_file_name + "_" + str(seed) + ".npy") for seed in seeds]
      
      sns.set_style("whitegrid")
      sns.tsplot(
        condition=plot_name, data=results, color=colors[c], linestyle='-', err_style='ci_band', ci=95, interpolate=True, legend=True)

    except:
      continue 

    c += 1

  sns.set_style("whitegrid")
  legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=0., title=env,prop={'size': 20})
  plt.setp(legend.get_title(),fontsize=30)
  
  plt.show()
  plt.pause(.000001)
  plt.rc('legend', fontsize=100)    # legend fontsize
  plt.ylabel("Average Episodic Reward",fontsize=30)
  plt.xlabel("Timesteps x 5000",fontsize=30)
  plt.savefig(env + ".png")

  print(f"{env}\n")
  input("Press enter to see next plot.")

