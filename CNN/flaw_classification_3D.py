import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import uuid
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from mynet import MyNet


# A function to evaluate test data within training period
def eval_during_train(inp, targ):
    output = model(inp)
    _, predict = torch.max(output.data, 1)
    total = output.size(0)
    correct = (predict == targ).sum()
    return (correct/total*100).item()


# Random seed
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.001

# Dataset dictionary
dataset_dict = {
    1: 'Type_1_1',
    2: 'Type_2_536',
    3: 'Type_3_529',
    4: 'Type_4_510',
    5: 'Type_5_509'
}

# Dataset dictionary
exp_dict = {
    0: ['No_flaw1', 0],
    1: ['No_flaw2', 0],
    2: ['No_flaw3', 0],
    3: ['No_flaw4', 0],
    4: ['Sample5', 2],
    5: ['Sample6', 2],
    6: ['Sample7', 2],
    7: ['Sample8', 3],
    8: ['Sample9', 3],
    9: ['Sample10', 3],
    10: ['Sample11', 4],
    11: ['Sample12', 4],
    12: ['Sample13', 4],
    13: ['Sample14', 1],
    14: ['Sample15', 1],
    15: ['Sample16', 1],
    16: ['Sample5_off', 2],
    17: ['Sample6_off', 2],
    18: ['Sample7_off', 2],
    19: ['Sample8_off', 3],
    20: ['Sample9_off', 3],
    21: ['Sample10_off', 3],
    22: ['Sample11_off', 4],
    23: ['Sample12_off', 4],
    24: ['Sample13_off', 4],
    25: ['Sample14_off', 1],
    26: ['Sample15_off', 1],
    27: ['Sample16_off', 1]
}

# Flaw classes
flaw_classes = ['NF', 'SC', 'TC', 'SW', 'CW']
class_num = len(flaw_classes)

# Read data from Excel (only when reshuffling the train/test data)
reshuffle = False

if reshuffle:
    # Some weird signals not to be included
    weird_signals = {
        2: [8, 18, 36, 87, 148, 175, 199, 235, 246, 252, 267, 275, 278, 282, 345, 346,
            349, 353, 358, 362, 365, 368, 410, 413, 416, 429, 432, 461, 464, 489, 508,
            521],
        3: [4, 15, 23, 37, 131, 148, 149, 171, 192, 276, 311, 329,
            330, 383, 421, 430, 478, 486, 504, 508, 527],
        4: [78],
        5: [43, 90, 139, 147, 195, 284, 334]
    }

    # Load the data, first copy the no flaw case 500 times
    temp1 = pd.read_excel('train/' + dataset_dict[1] + '.xlsx', header=None).values
    temp2 = pd.read_excel('train/' + dataset_dict[1] + '_info.xlsx', header=None).values
    signals = np.tile(temp1, (500, 1))
    signals_info = np.tile(temp2, 500)

    for i in range(2, 6):
        # Read signal and info file
        temp1 = pd.read_excel('train/' + dataset_dict[i] + '.xlsx', header=None).values
        temp2 = pd.read_excel('train/' + dataset_dict[i] + '_info.xlsx', header=None).values

        # Remove nan rows from signals
        temp1 = temp1[~np.isnan(temp1).any(axis=1)]
        # Remove weird signals by rows
        temp1 = np.delete(temp1, weird_signals[i], 0)

        # Remove nan columns from info
        temp2 = temp2[:, ~np.isnan(temp2).any(axis=0)]
        # Remove weird signals info by columns
        temp2 = np.delete(temp2, weird_signals[i], 1)

        # Combine data together
        signals = np.concatenate([signals, temp1[:500, :]])
        signals_info = np.concatenate([signals_info, temp2[:, :500]], axis=1)

    # Format the data
    inputs = abs(signals[:, 601:]/max(signals[0, :]))
    targets = signals_info[0, :] - 1

    # Divide data into training set and testing set
    # 80% are used for training, and 20% for testing
    train_index = [i for i in range(400)] + [i for i in range(500, 900)] + [i for i in range(1000, 1400)] +\
                                            [i for i in range(1500, 1900)] + [i for i in range(2000, 2400)]
    test_index = list(set(range(inputs.shape[0])) - set(train_index))
    inputs_train, inputs_test = inputs[train_index], inputs[test_index]
    targets_train, targets_test = targets[train_index], targets[test_index]

    np.save('train/inputs_train.npy', inputs_train)
    np.save('train/inputs_test.npy', inputs_test)
    np.save('train/targets_train.npy', targets_train)
    np.save('train/targets_test.npy', targets_test)

else:
    inputs_train = np.load('train/inputs_train.npy')
    inputs_test = np.load('train/inputs_test.npy')
    targets_train = np.load('train/targets_train.npy')
    targets_test = np.load('train/targets_test.npy')

# From numpy array to torch tensor
# Additional notes: the inputs are changed from float64 to float32
# Additional notes: the targets are changed into long type (int64)
inputs_train = torch.unsqueeze(torch.from_numpy(inputs_train).float(), 1)
inputs_test = torch.unsqueeze(torch.from_numpy(inputs_test).float(), 1)
targets_train = torch.from_numpy(targets_train).long()
targets_test = torch.from_numpy(targets_test).long()

# Load experimental data
inputs_exp = np.ndarray([len(exp_dict), 4001])
targets_exp = np.ndarray([len(exp_dict)])

for i in exp_dict:
    inputs_exp[i, :] = np.genfromtxt('exp/' + exp_dict[i][0] + '.txt', delimiter=',', dtype=np.float32)
    targets_exp[i] = exp_dict[i][1]

inputs_exp = torch.unsqueeze(torch.from_numpy(abs(inputs_exp[:, 601:] / 1.043936254833034e-09)).float(), 1)
targets_exp = torch.from_numpy(targets_exp).long()

# CNN model
model = MyNet()
model.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Store loss for plotting
loss_tr = []

# Define correct/total ratio for performance evaluation
class_correct_test = list(0. for item in range(class_num))
class_total_test = list(0. for item in range(class_num))
class_correct_exp = list(0. for item in range(class_num))
class_total_exp = list(0. for item in range(class_num))
test_performance = []
train_performance = []
exp_performance = []

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs_train = model(inputs_train)
    loss = criterion(outputs_train, targets_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_tr.append(loss.item())

    # Pass the testing data
    model.eval()
    test_performance.append(eval_during_train(inputs_test, targets_test))
    train_performance.append(eval_during_train(inputs_train, targets_train))
    exp_performance.append(eval_during_train(inputs_exp, targets_exp))
    model.train()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch + 1, num_epochs, loss.item()))

# Do a final test and experimental data evaluation
model.eval()
with torch.no_grad():
    # For test
    outputs_test = model(inputs_test)
    _, predicts_test = torch.max(outputs_test.data, 1)
    total = outputs_test.size(0)
    correct = (predicts_test == targets_test).sum()
    test_overall = correct / total * 100
    for i in range(class_num):
        indices = [j for j, x in enumerate(targets_test) if x == i]
        class_total_test[i] = len(indices)
        class_correct_test[i] = (predicts_test[indices] == i).sum().item()

    # For experiments
    outputs_exp = model(inputs_exp)
    _, predicts_exp = torch.max(outputs_exp.data, 1)
    total = outputs_exp.size(0)
    correct = (predicts_exp == targets_exp).sum()
    exp_overall = correct / total * 100
    for i in range(class_num):
        indices = [j for j, x in enumerate(targets_exp) if x == i]
        class_total_exp[i] = len(indices)
        class_correct_exp[i] = (predicts_exp[indices] == i).sum()

# Generate a random path and directory to store temp-trained model
temp_name = str(uuid.uuid4())[-8:] + '/'
model_path = 'temp_model/' + temp_name
os.mkdir(model_path)

# Save the model
torch.save(model, model_path + 'my_model.pt')

# Write the model parameters to file
with open(model_path + 'model_param.txt', 'w') as f:
    # Writing data to a file
    f.write('---------------------------------------------------------------')
    f.write('\n Training epochs: ' + str(num_epochs))
    f.write('\n Learning rate: ' + str(learning_rate))
    f.write('\n---------------------------------------------------------------')
    f.write('\n CNN configuration:')
    f.write('\n' + str(model))
    f.write('\n---------------------------------------------------------------')

# Print out performance
print('----------------Testing data-------------------')
print(confusion_matrix(targets_test, predicts_test))
print('Total classification accuracy is: %.2f %%' % (test_overall))
for i in range(class_num):
    print('Accuracy of %2s : %.2f %%' % (flaw_classes[i], class_correct_test[i] / class_total_test[i] * 100))
print('------------------------------------------------')

print('----------------Experiment data-----------------')
print('Total classification accuracy is: %.2f %%' % (exp_overall))
for i in range(class_num):
    if class_total_exp[i] != 0:
        print('Accuracy of %2s : %.2f %%' % (
            flaw_classes[i], 100 * class_correct_exp[i] / class_total_exp[i]))
    else:
        print('Accuracy of %2s is NA' % (flaw_classes[i]))
print(confusion_matrix(targets_exp, predicts_exp))
print('------------------------------------------------')

# Plot and save loss into npy file
fig, ax = plt.subplots(1, 1)
ax.plot(loss_tr)
ax.set_xlabel('Epochs')
ax.set_yscale('log')
ax.set_title('Training loss')
ax.grid(linestyle='--')
fig.show()
np.save(model_path + 'loss.npy', np.array(loss_tr))

# Plot and save the test data performance evolution
fig, ax = plt.subplots(1, 1)
ax.plot(train_performance)
ax.plot(test_performance)
ax.plot(exp_performance)
ax.legend(['Training data', 'Testing data', 'Experimental data'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Classification rate (%)')
ax.grid(linestyle='--')
fig.show()
np.save(model_path + 'test_performance.npy', np.array(test_performance))
np.save(model_path + 'train_performance.npy', np.array(train_performance))
np.save(model_path + 'exp_performance.npy', np.array(exp_performance))

# Plot the experiment data confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(targets_exp, predicts_exp, display_labels=flaw_classes,
                                        normalize='true', values_format='.0%', cmap='GnBu')
disp.im_.colorbar.ax.set_yticklabels(['0', '20%', '40%', '60%', '80%', '100%'])
disp.ax_.set_xlabel('Predicted flaw categories')
disp.ax_.set_ylabel('True flaw categories')
plt.show()

# Print the temporary file name
print('Temporary file name is: ' + temp_name)
