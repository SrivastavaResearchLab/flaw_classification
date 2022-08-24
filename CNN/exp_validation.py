import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pretty_confusion_matrix import pp_matrix_from_data
from mynet import MyNet

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

plt.rcParams["font.family"] = "Times New Roman"

# Flaw classes
# 0 for no crack; 1 for single crack; 2 for two cracks;
# 3 for single wall loss; 4 for crack and wall loss
flaw_classes = ['NF', 'SC', 'SW', 'TC', 'CW']
class_num = len(flaw_classes)

# Dataset dictionary
exp_dict = {
    0: ['No_flaw1', 0],
    #1: ['No_flaw2', 0],
    #2: ['No_flaw3', 0],
    #3: ['No_flaw4', 0],
    1: ['Sample5', 2],
    2: ['Sample6', 2],
    3: ['Sample7', 2],
    4: ['Sample8', 3],
    5: ['Sample9', 3],
    6: ['Sample10', 3],
    7: ['Sample11', 4],
    8: ['Sample12', 4],
    9: ['Sample13', 4],
    10: ['Sample14', 1],
    11: ['Sample15', 1],
    12: ['Sample16', 1],
    13: ['Sample5_off', 2],
    14: ['Sample6_off', 2],
    15: ['Sample7_off', 2],
    16: ['Sample8_off', 3],
    17: ['Sample9_off', 3],
    18: ['Sample10_off', 3],
    19: ['Sample11_off', 4],
    20: ['Sample12_off', 4],
    21: ['Sample13_off', 4],
    22: ['Sample14_off', 1],
    23: ['Sample15_off', 1],
    24: ['Sample16_off', 1]
}

# Load experimental data
inputs_exp = np.ndarray([len(exp_dict), 4001])
targets_exp = np.ndarray([len(exp_dict)])

for i in exp_dict:
    inputs_exp[i, :] = np.genfromtxt('exp/' + exp_dict[i][0] + '.txt', delimiter=',', dtype=np.float32)
    targets_exp[i] = exp_dict[i][1]

inputs_exp = torch.unsqueeze(torch.from_numpy(abs(inputs_exp[:, 601:]/1.043936254833034e-09)).float(), 1)
targets_exp = torch.from_numpy(targets_exp).long()

# Load the trained CNN
model = torch.load('saved_model/main/my_model.pt')
model.eval()

class_correct = list(0. for i in range(class_num))
class_total = list(0. for i in range(class_num))

# Test model and error analysis
with torch.no_grad():
    outputs_exp= model(inputs_exp)
    _, predicts_exp = torch.max(outputs_exp.data, 1)
    total = outputs_exp.size(0)
    correct = (predicts_exp == targets_exp).sum()
    for i in range(class_num):
        indices = [j for j, x in enumerate(targets_exp) if x == i]
        class_total[i] = len(indices)
        class_correct[i] = (predicts_exp[indices] == i).sum()

# Switch the index 2 and 3 for TC and SW
targets_exp[targets_exp == 2] = 5
targets_exp[targets_exp == 3] = 2
targets_exp[targets_exp == 5] = 3
predicts_exp[predicts_exp == 2] = 5
predicts_exp[predicts_exp == 3] = 2
predicts_exp[predicts_exp == 5] = 3
class_correct[2], class_correct[3] = class_correct[3], class_correct[2]

print('Total classification accuracy is: %.2f %%' % (100 * correct / total))
for i in range(class_num):
    if class_total[i] != 0:
        print('Accuracy of %2s : %.2f %%' % (
            flaw_classes[i], 100 * class_correct[i] / class_total[i]))
    else:
        print('Accuracy of %2s is NA' % (flaw_classes[i]))

# Plot confusion  matrix
plt.rcParams['font.size'] = '14'
disp = ConfusionMatrixDisplay.from_predictions(targets_exp, predicts_exp, display_labels=flaw_classes,
                                        normalize='true', values_format='.0%', cmap='GnBu')
disp.im_.colorbar.ax.set_yticklabels(['0', '20%', '40%', '60%', '80%', '100%'])
disp.ax_.set_xlabel('Predicted flaw categories')
disp.ax_.set_ylabel('True flaw categories')
#plt.show()
plt.savefig('Conf_mat_exp.eps', dpi=150)

# Performance plot
plt.rcParams['font.size'] = '45'
exp_performance_main = np.load('saved_model/main/exp_performance.npy')
exp_performance_nodo = np.load('saved_model/nodo/test_performance.npy')
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.plot(exp_performance_main, 'k', linewidth=3)
ax.plot(exp_performance_nodo, 'k--', linewidth=3)
ax.legend(['With dropout', 'Without dropout'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Overall accuracy (%)')
ax.grid(linestyle='--')
#plt.show()
plt.savefig('Accuracy_exp.eps', dpi=300)
