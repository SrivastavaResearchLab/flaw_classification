import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from mynet import MyNet

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

plt.rcParams['font.size'] = '45'
plt.rcParams["font.family"] = "Times New Roman"

# Load testing data
inputs_test = np.load('train/inputs_test.npy')
targets_test = np.load('train/targets_test.npy')

# Flaw classes
# 0 for no crack; 1 for single crack; 2 for two cracks;
# 3 for single wall loss; 4 for crack and wall loss
flaw_classes = ['NF', 'SC', 'SW', 'TC', 'CW']
class_num = len(flaw_classes)

# From numpy array to torch tensor
inputs_test = torch.unsqueeze(torch.from_numpy(inputs_test).float(), 1)
targets_test = torch.from_numpy(targets_test).long()

# Load the trained CNN
model = torch.load('temp_model/my_model.pt')
model.eval()

# Define correct/total ratio for performance evaluation
class_correct = list(0. for item in range(class_num))
class_total = list(0. for item in range(class_num))

# Do a test data evaluation
with torch.no_grad():
    outputs_test = model(inputs_test)
    _, predicts_test = torch.max(outputs_test.data, 1)
    total = outputs_test.size(0)
    correct = (predicts_test == targets_test).sum()
    for i in range(class_num):
        indices = [j for j, x in enumerate(targets_test) if x == i]
        class_total[i] = len(indices)
        class_correct[i] = (predicts_test[indices] == i).sum().item()

# Loss plot
loss = np.load('temp_model/loss.npy')
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.plot(loss)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(linestyle='--')
plt.savefig('Loss.eps', dpi=300)

# Performance plot
train_performance = np.load('temp_model/train_performance.npy')
test_performance = np.load('temp_model/test_performance.npy')
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.plot(train_performance, 'k', linewidth=3)
ax.plot(test_performance, 'k--', linewidth=3)
ax.legend(['Training data', 'Testing data'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Overall accuracy (%)')
ax.grid(linestyle='--')
plt.savefig('Accuracy_test.eps', dpi=300)

# Switch the index 2 and 3 for TC and SW
targets_test[targets_test == 2] = 5
targets_test[targets_test == 3] = 2
targets_test[targets_test == 5] = 3
predicts_test[predicts_test == 2] = 5
predicts_test[predicts_test == 3] = 2
predicts_test[predicts_test == 5] = 3
class_correct[2], class_correct[3] = class_correct[3], class_correct[2]

# Confusion matrix plot
plt.rcParams['font.size'] = '18'
disp = ConfusionMatrixDisplay.from_predictions(targets_test, predicts_test, display_labels=flaw_classes,
                                        normalize='true', values_format='.0%', cmap='GnBu')
disp.im_.colorbar.ax.set_yticklabels(['0', '20%', '40%', '60%', '80%', '100%'])
disp.ax_.set_xlabel('Predicted flaw categories')
disp.ax_.set_ylabel('True flaw categories')
plt.savefig('Conf_mat_test.eps', dpi=150)

print(confusion_matrix(targets_test, predicts_test))
print('Total classification accuracy is: %.2f %%' % (correct/total*100))
for i in range(class_num):
    print('Accuracy of %2s : %.2f %%' % (flaw_classes[i], class_correct[i]/class_total[i]*100))
