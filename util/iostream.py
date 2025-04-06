from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import utc, timezone
from util.time import timestamp2str
import json
import argparse
import numpy as np

def printsep():
    print('='*40+'\n')

def save_attack_infos(f1_scores, total_err_scores, labels, names, save_path, dataset, config):
    slide_win=config['slide_win']
    down_len=config['down_len']

    
    if dataset == 'wadi' or dataset == 'wadi2':
        s = '09/10/2017 18:00:00'
    elif dataset == 'swat':
        s = '28/12/2015 10:00:00'
    start_s = int(time.mktime(datetime.strptime(s, "%d/%m/%Y %H:%M:%S").timetuple()))
    cst8 = timezone('Asia/Shanghai')
    fmt = '%m/%d %H:%M:%S'

    attack_inters = get_attack_interval(labels)

    save_infos = {
        'total_best_f1_score': f1_scores[0],
        'total_best_f1_score_topk': f1_scores[1],
        'total_best_f1_score_all': f1_scores[2],
        'attacks': []
    }

    indices_map = names

    indices = np.argmax(total_err_scores, axis=0).tolist()
    anomaly_sensors = [ indices_map[index] for index in indices ]

    topk = 5
    topk_indices = np.argpartition(total_err_scores, -topk, axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)

    topk_anomaly_sensors = []
    topk_err_score_map=[]
    for i, indexs in enumerate(topk_indices):
        # print(indexs)
        topk_anomaly_sensors.append([indices_map[index] for index in indexs])

        item = {}
        for sensor, index in zip(topk_anomaly_sensors[i],indexs):
            if sensor not in item:
                item[sensor] = total_err_scores[index, i]

        topk_err_score_map.append(item)

        
    for head, end in attack_inters:
        attack_infos = {}
        topk_attack_infos = {}

        head_t = timestamp2str(start_s+(head+slide_win)*down_len, fmt, cst8)
        end_t = timestamp2str(start_s+(end+slide_win)*down_len, fmt, cst8)
        # head_t = datetime.fromtimestamp(start_s+head).astimezone(cst8).strftime(fmt)
        # end_t = datetime.fromtimestamp(start_s+end).astimezone(cst8).strftime(fmt)

        # print(f'\nattack from {head_t} to {end_t}:')
        
        for i in range(head, end):
            # t = datetime.fromtimestamp(start_s+i).astimezone(cst8).strftime(fmt)
            t = timestamp2str(start_s+(i+slide_win)*down_len, fmt, cst8)
            max_sensor = anomaly_sensors[i]
            topk_sensors = topk_anomaly_sensors[i]

            if max_sensor not in attack_infos:
                attack_infos[max_sensor] = 0
            attack_infos[max_sensor] += 1

            # for anomaly_sensor in topk_sensors:
            #     if anomaly_sensor not in topk_attack_infos:
            #         topk_attack_infos[anomaly_sensor] = 0
            #     topk_attack_infos[anomaly_sensor] += 1
            
            for anomaly_sensor in topk_sensors:
                if anomaly_sensor not in topk_attack_infos:
                    topk_attack_infos[anomaly_sensor] = 0
                topk_attack_infos[anomaly_sensor] += topk_err_score_map[i][anomaly_sensor]
            

        # print('-------------------------------')
        # print(f'total top 5 attack sensors from {head_t} to {end_t}:')
        sorted_attack_infos = {k: v for k, v in sorted(attack_infos.items(), reverse=True, key=lambda item: item[1])}
        sorted_topk_attack_infos = {k: v for k, v in sorted(topk_attack_infos.items(), reverse=True, key=lambda item: item[1])}
        # for key, count in sorted_attack_infos.items()[:5]:
        #     print(key, count)

        save_infos['attacks'].append({
            'start': head_t,
            'end': end_t,
            'sensors': list(sorted_attack_infos),
            'topk_sensors': list(sorted_topk_attack_infos),
            'topk_scores': list(sorted_topk_attack_infos.values())
        })

    with open(save_path, 'w+') as outfile:
        json.dump(save_infos, outfile, indent=4)  

def export_fn_fp_timesteps(pred_labels, gt_labels, dataset_name, slide_win):
    """
    Export false negative and false positive timesteps to separate files.
    
    Args:
        total_err_scores: The anomaly scores for each time point
        gt_labels: Ground truth labels (0: normal, 1: anomaly)
        dataset_name: Name of the dataset (for filename)
        slide_win: Sliding window size from the configuration
    """
    import numpy as np
    from datetime import datetime
    import time
    
    # TODO to use
    gt_labels = np.array(gt_labels)
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])
    
    # Find false negatives (gt=1, pred=0) and false positives (gt=0, pred=1)
    fn_indices = np.where((gt_labels == 1) & (pred_labels == 0))[0]
    fp_indices = np.where((gt_labels == 0) & (pred_labels == 1))[0]
    
    # Set start timestamp based on dataset
    if dataset_name == 'wadi' or dataset_name == 'wadi2':
        start_s = int(time.mktime(datetime.strptime('09/10/2017 18:00:00', "%d/%m/%Y %H:%M:%S").timetuple()))
    elif dataset_name == 'swat':
        start_s = int(time.mktime(datetime.strptime('28/12/2015 10:00:00', "%d/%m/%Y %H:%M:%S").timetuple()))
    else:
        # Default to current time if dataset is unknown
        start_s = int(time.time())
    
    # Calculate timesteps - adding slide_win to account for the sliding window offset
    down_len = 1  # Default downsampling length is 1 unless specified otherwise
    
    # Export false negatives
    with open(f'./results/{dataset_name}/fn_timesteps.txt', 'w') as f:
        for idx in fn_indices:
            timestep = start_s + (idx + slide_win) * down_len
            timestamp = datetime.fromtimestamp(timestep).strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{idx},{timestamp}\n")
    
    # Export false positives
    with open(f'./results/{dataset_name}/fp_timesteps.txt', 'w') as f:
        for idx in fp_indices:
            timestep = start_s + (idx + slide_win) * down_len
            timestamp = datetime.fromtimestamp(timestep).strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{idx},{timestamp}\n")
    
    print(f"Exported {len(fn_indices)} false negatives to ./results/{dataset_name}/fn_timesteps.txt")
    print(f"Exported {len(fp_indices)} false positives to ./results/{dataset_name}/fp_timesteps.txt")


# To integrate this with the existing code, add this to the get_score method in main.py:
# After this line in get_score:
#     print(f'F1 score: {info[0]}')
#     print(f'precision: {info[1]}')
#     print(f'recall: {info[2]}\n')
#
# Add:
#     export_fn_fp_timesteps(test_scores, test_labels, info[4], self.env_config['dataset'], self.train_config['slide_win'])
