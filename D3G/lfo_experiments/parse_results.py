import numpy as np 

def average_score(file_name):
  score = 0.
  all_scores = [] 

  for seed in range(10):
    file_name_seed = f"{file_name}_{seed}.txt"
    max_score = None
 
    try: 
      f = open(file_name_seed, "r")
      
      for line in f:
        step, score = (line.replace("\n", "")).split(" ")
        if max_score is None:
          max_score = float(score)
        else:
          max_score = max(float(score), max_score)

      if max_score is not None:
        all_scores.append(max_score)

    except Exception as e:
      print(e)
      continue
   
  print(all_scores) 
  return f"{np.mean(all_scores)} +- {np.std(all_scores)}"

if __name__ == "__main__":
  for env in ["Reacher-v2", "InvertedPendulum-v2"]:
    print(f"Environment {env}")
    for randomness in [0.0,0.25,0.5,0.75,1.0]:
      bco_file_name = f"bco_results/D3G_{env}_{randomness}_{True}" 
      d3g_file_name = f"bco_results/D3G_{env}_{randomness}_{False}" 
      bco_score = average_score(bco_file_name)
      d3g_score = average_score(d3g_file_name)
      print(f"Randomness: {randomness} BCO: {bco_score} D3G: {d3g_score}")

    print("\n")
