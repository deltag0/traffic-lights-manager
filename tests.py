import os

import matplotlib.pyplot as plt

from agents import DQN_Agent, PPO_Agent, Standard_Cycle
from constants import DIR_PATH


model = PPO_Agent("2x2")
evaluator = Standard_Cycle("4_way")

results_file = os.path.join(DIR_PATH, "results", "comparison_4x4_V4.png")

model_results = model.run(False)
evaluator_results = evaluator.run()

combined_results = list(zip(model_results, evaluator_results))
len_results = len(combined_results)
better_performance_count = 0

for model_result, evaluator_result in combined_results:
    if model_result < evaluator_result:
        better_performance_count += 1

performance_percent = better_performance_count / len_results * 100

print(f"The model did better than the standard cycle {performance_percent}% of the time")

fig = plt.figure(1)

plt.plot(range(len_results), model_results, label="Model Results")
plt.plot(range(len_results), evaluator_results, label="Standard Results")

plt.ylabel("Mean waiting time (s)")
plt.xlabel("Number of steps")

plt.subplots_adjust(wspace=1.0, hspace=1.0)

plt.legend()
fig.savefig(results_file)
plt.close(fig)
