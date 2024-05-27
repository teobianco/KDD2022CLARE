from utils.metrics import get_nmi_score

dataset = 'Amazon_setting1'

# Calculate STABILITY as the ONMI score between subsequent timesteps in dataset communities
data_path = f'./dataset/{dataset}/'
save_path = f'./dataset/{dataset}/onmi_transitions.txt'
first_time_path = data_path + f'time_0/{dataset}_0-1.90.cmty.txt'
first_time_comm = []
with open(first_time_path, 'r') as f:
    for line in f:
        first_time_comm.append(list(map(int, line.split())))
onmi_scores = []
for i in range(9):
    print(f'STARTING CONFRONTING TIMESTEP {i} AND {i+1} ...')
    second_time_path = data_path + f'time_{i+1}/{dataset}_{i+1}-1.90.cmty.txt'
    second_time_comm = []
    with open(second_time_path, 'r') as f:
        for line in f:
            second_time_comm.append(list(map(int, line.split())))
    onmi_scores.append(get_nmi_score(first_time_comm, second_time_comm))
    first_time_comm = second_time_comm

print('Scores in every transition', onmi_scores)
print('Mean score:', sum(onmi_scores) / len(onmi_scores))
# Save the scores in save_path
with open(save_path, 'w') as f:
    for score in onmi_scores:
        f.write(f'{score}\n')
    f.write(f'Mean score: {sum(onmi_scores) / len(onmi_scores)}')
