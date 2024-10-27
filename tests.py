from agents import DQN_Agent, PPO_Agent, Standard_Cycle


model = PPO_Agent("TwoxTwo")
evaluator = Standard_Cycle("TwoxTwo")

model_results = model.run(False)
evaluator_results = Standard_Cycle.run()

combined_results = list(zip(model_results, evaluator_results))
len_results = len(combined_results)
better_performance_count = 0

for model_result, evaluator_result in combined_results:
    if model_result > evaluator_result:
        better_performance_count += 1

performance_percent = better_performance_count / len_results * 100

print(f"The model did better than the standard cycle {performance_percent}% of the time")
