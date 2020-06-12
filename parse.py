import numpy as np
from sklearn.utils import shuffle

def img_data_to_csv(data, labels, label_names, out_file_str):
    
    assert len(labels) == len(data)
    
    mean = np.zeros(len(data[0]), dtype=np.float32)
    std = np.zeros(len(data[0]), dtype=np.float32)
    
    for k in range(len(data)):
        mean += data[k]
    mean = mean / len(data)
    for k in range(len(data)):
        std += (data[k] - mean) ** 2
    std = std / len(data)
    std = np.sqrt(std + 0.000001)
    
    for k in range(len(data)):
        data[k] = np.array(data[k], dtype=np.float32)
        data[k] = (data[k] - mean) / std
    data, labels = shuffle(data, labels)
    
    with open(out_file_str, "w+") as f:
        mean_line = [str(k) for k in mean]
        std_line = [str(k) for k in std]
        f.write(str(len(data)) + ',')
        f.write(str(len(data[0])) + ',')
        f.write(str(len(label_names)) + ',')
        f.write(','.join(label_names) + '\n')
        f.write(','.join(mean_line) + '\n')
        f.write(','.join(std_line) + '\n')
        k = 0
        for i in range(len(data)):
            line = [str(k) for k in data[i]]
            f.write(','.join(line) + ',')
            f.write(str(labels[k]) + '\n')
            k += 1