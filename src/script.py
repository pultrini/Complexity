# %%
import subprocess
import json
import os
import math

#%%
def run_script(script):
    result = subprocess.run(['python3', script], capture_output=True, text=True)
    if result.returncode !=0:
        print(f'Erro ao executar {script}: {result.stderr}')
        return None
    else:
        print(f'{script} executado com sucesso')
        return result.stdout

def read_metrics(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            metrics = json.load(file)
        return metrics
    else:
        print(f"Arquivo {file_path} não encontrado.")
        return None

iteration = 1
max_iteration=5

all_metrics={}

output_file = 'results/metrics/all_metrics.json'

while iteration <= max_iteration:
    run_script('src/model1.py')
    run_script('src/model2.py')
    run_script('src/model3.py')


    metrics_file_1 = 'results/metrics/metrics_model1.json'
    metrics_file_2 = 'results/metrics/metrics_model2.json'
    metrics_file_3 = 'results/metrics/metrics_model3.json'

    metrics1 = read_metrics(metrics_file_1)
    metrics2 = read_metrics(metrics_file_2)
    metrics3 = read_metrics(metrics_file_3)

    if metrics1 and metrics2 and metrics3:
        all_metrics[f'{iteration}'] = {
            'model1' : metrics1,
            'model2 (complexity)' : metrics2,
            'model3 (loss)' : metrics3
        }
        with open(output_file, 'w') as file:
            json.dump(all_metrics, file, indent=4)

        print("Metricas salvas")
    iteration +=1

# %%
import json
import math
with open('results/metrics/all_metrics.json', 'r') as file:
    data = json.load(file)


sum_accuracy = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}
sum_loss = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}
sum_complexity = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}
num_rounds = len(data)


for round_data in data.values():
    for model in ['model1', 'model2 (complexity)', 'model3 (loss)']:
        sum_accuracy[model] += round_data[model]['Max_accuracy']
        sum_loss[model] += round_data[model]['Min_loss']
        sum_complexity[model] += round_data[model]['Max_complexity']

avg_accuracy = {model: sum_accuracy[model]/num_rounds for model in sum_accuracy}
avg_loss = {model: sum_loss[model]/num_rounds for model in sum_loss}
avg_complexity = {model: sum_complexity[model]/num_rounds for model in sum_complexity}


sum_sq_accuracy = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}
sum_sq_loss = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}
sum_sq_complexity = {'model1': 0, 'model2 (complexity)': 0, 'model3 (loss)': 0}

for round_data in data.values():
    for model in ['model1', 'model2 (complexity)', 'model3 (loss)']:
        sum_sq_accuracy[model] += (round_data[model].get('Max_accuracy', 0) - avg_accuracy[model])**2
        sum_sq_loss[model] += (round_data[model].get('Max_loss', 0) - avg_loss[model])**2
        sum_sq_complexity[model] += (round_data[model].get('Max_complexity', 0) - avg_complexity[model])**2

std_dev_accuracy = {model: math.sqrt(sum_sq_accuracy[model]/num_rounds) for model in sum_accuracy}
std_dev_loss = {model: math.sqrt(sum_sq_loss[model]/num_rounds) for model in sum_loss}
std_dev_complexity = {model: math.sqrt(sum_sq_complexity[model]/num_rounds) for model in sum_complexity}

results = {}

for model in ['model1', 'model2 (complexity)', 'model3 (loss)']:
    results[model] = {
        'Max_accuracy': f"{avg_accuracy[model]:.5f} +- {std_dev_accuracy[model]:.5f}",
        'Min_loss': f"{avg_loss[model]:.5f} +- {std_dev_loss[model]:.5f}",
        'Max_complexity': f"{avg_complexity[model]:.5f} +- {std_dev_complexity[model]:.5f}"
    }
with open('results/results.json', 'w') as file:
    json.dump(results,file, indent = 4)


# %%
