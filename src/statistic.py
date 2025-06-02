import json
import math

with open('all_metrics.json', 'r') as file:
    data = json.load(file)


sum_accuracy = {'model1': 0, 'model2': 0, 'model3': 0}
sum_loss = {'model1': 0, 'model2': 0, 'model3': 0}
sum_complexity = {'model1': 0, 'model2': 0, 'model3': 0}
num_rounds = len(data)


for round_data in data.values():
    for model in ['model1', 'model2', 'model3']:
        sum_accuracy[model] += round_data[model]['Max_accuracy']
        sum_loss[model] += round_data[model]['Min_loss']
        sum_complexity[model] += round_data[model]['Max_complexity']

avg_accuracy = {model: sum_accuracy[model]/num_rounds for model in sum_accuracy}
avg_loss = {model: sum_loss[model]/num_rounds for model in sum_loss}
avg_complexity = {model: sum_complexity[model]/num_rounds for model in sum_complexity}

sum_sq_accuracy = {'model1': 0, 'model2': 0, 'model3': 0}
sum_sq_loss = {'model1': 0, 'model2': 0, 'model3': 0}
sum_sq_complexity = {'model1': 0, 'model2': 0, 'model3': 0}

for round_data in data.values():
    for model in ['model1', 'model2', 'model3']:
        sum_sq_accuracy[model] += (round_data[model].get('Max_accuracy', 0) - avg_accuracy[model])**2
        sum_sq_loss[model] += (round_data[model].get('Max_loss', 0) - avg_loss[model])**2
        sum_sq_complexity[model] += (round_data[model].get('Max_complexity', 0) - avg_complexity[model])**2

std_dev_accuracy = {model: math.sqrt(sum_sq_accuracy[model]/num_rounds) for model in sum_accuracy}
std_dev_loss = {model: math.sqrt(sum_sq_loss[model]/num_rounds) for model in sum_loss}
std_dev_complexity = {model: math.sqrt(sum_sq_complexity[model]/num_rounds) for model in sum_complexity}



results = {}

for model in ['model1', 'model2', 'model3']:
    results[model] = {
          'Max_accuracy': f"{avg_accuracy[model]:.5f} +- {std_dev_accuracy[model]:.5f}",
        'Min_loss': f"{avg_loss[model]:.5f} +- {std_dev_loss[model]:.5f}",
        'Max_complexity': f"{avg_complexity[model]:.5f} +- {std_dev_complexity[model]:.5f}"
    }
with open('results.json', 'w') as file:
    json.dump(results,file, indent = 4)
