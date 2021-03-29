import pickle
import torch
import torch.nn.functional as F
from model import graph_conv
from data_preparation import data_setup
from sklearn.metrics import precision_recall_fscore_support
from early_stopping import EarlyStopping

# Loading the data
infile_train = open('data/data_train.pkl', 'rb')
data_train = pickle.load(infile_train)

infile_val = open('data/data_val.pkl', 'rb')
data_val = pickle.load(infile_val)

infile_test = open('data/data_test.pkl', 'rb')
data_test = pickle.load(infile_test)

edge_type = 'undirected'

edges_list_tensors, vertices_list_tensors, idx_list_tensors, target = data_setup(data_train, edge_type)    
edges_val, vertices_val, idx_val, target_val = data_setup(data_val, edge_type)
edges_test, vertices_test, idx_test, target_test = data_setup(data_test, edge_type)  


embedding_size = len(vertices_list_tensors[0][0])
hidden_dim = 125
hidden_dim2 = 100
learning_rate = 0.001
dropout = 0.2

device = torch.device('cpu')
model = graph_conv(embedding_size, hidden_dim, hidden_dim2, dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

samples = len(edges_list_tensors)
results_all = []
loss_all = []
epoch_acc_all = []
epoch_loss_all = []

epoch_acc_all_val = []
epoch_loss_all_val = []
results_all_val = []

# number of epochs
num_epochs = 100

# early stopping patience
patience = 5
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True, path='checkpoint.pt')

for epoch in range(num_epochs): 
    loss_sum = 0
    epoch_acc = 0
    results_epoch = []
    
    loss_sum_val = 0
    epoch_acc_val = 0
    results_epoch_val = []
    
    model.train()
    optimizer.zero_grad()
    for i in range(samples):
    
        #optimizer.zero_grad()

        result = model(edges_list_tensors[i], vertices_list_tensors[i], idx_list_tensors[i])
        results_all.append(result)
        results_epoch.append(result)
    
        loss = F.cross_entropy(result, target[i].view(-1))
        loss_all.append(loss)
        loss_sum += loss
    
    pred_labels = []
    for j in results_epoch:
        predicted_prob, pred_ind = torch.max(F.softmax(j, dim=-1), 1)
        pred_labels.append(pred_ind)
    
    correct_count = 0
    for i in range(len(target)):
        if pred_labels[i] == target[i]:
            correct_count += 1
    
    epoch_acc = correct_count/len(target)
    epoch_acc_all.append(epoch_acc)
    epoch_loss_all.append(loss_sum.item()/samples)
    loss_sum.backward()
    optimizer.step()
    
    model.eval()
    #with torch.no_grad():
    
    for i in range(len(edges_val)):

        result_val = model(edges_val[i], vertices_val[i], idx_val[i])
        results_all_val.append(result_val)
        results_epoch_val.append(result_val)
            
        loss_val = F.cross_entropy(result_val, target_val[i].view(-1))
        loss_sum_val += loss_val
            #acc = binary_accuracy(predictions, batch.label)
            #epoch_val_acc += acc.item()
           
    pred_labels_val = []
    for j in results_epoch_val:
        predicted_prob_val, pred_ind_val = torch.max(F.softmax(j, dim=-1), 1)
        pred_labels_val.append(pred_ind_val)
            
    correct_count_val = 0
    for i in range(len(target_val)):
        if pred_labels_val[i] == target_val[i]:
            correct_count_val += 1
                
    epoch_acc_val = correct_count_val/len(target_val)
    epoch_acc_all_val.append(epoch_acc_val)
    epoch_loss_all_val.append(loss_sum_val.item()/len(target_val))
    
    print('Epoch {:04d} | '.format(epoch) + ' Avg Epoch Loss: {:.4f} | '.format(loss_sum.item()/samples) + ' Validation Loss: {:.4f} | '.format(loss_sum_val.item()/len(target_val)) + 
          ' Epoch Acc: {:.4f} | '.format(epoch_acc) + ' Validation Acc: {:.4f} | '.format(epoch_acc_val))
    
    valid_loss = loss_sum_val.item()/len(target_val)
    early_stopping(valid_loss, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

'''
Testing the trained model
'''

test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))


model.eval()
results_all_test = []
loss_test = []
loss_sum_test = 0
for i in range(len(data_test)):

    result_test = model(edges_test[i], vertices_test[i], idx_test[i])
    results_all_test.append(result_test)
    
    loss_t = F.cross_entropy(result_test, target_test[i].view(-1))
    loss_test.append(loss_t)
    loss_sum_test += loss_t
   
    predictions_test = []
    for j in results_all_test:
        _, pred_ind_test = torch.max(F.softmax(j, dim=-1), 1)
        predictions_test.append(pred_ind_test)
        

correct_test = 0
for i in range(len(data_test)):
    if predictions_test[i] == target_test[i]:
        correct_test += 1
            
acc_test = correct_test/len(data_test)

loss_test_avg = loss_sum_test/len(data_test)
    
 
correct_positive = 0
correct_negative = 0
wrong = 0
for i in range(len(data_test)):    
    if predictions_test[i] == target_test[i] == 1:
        correct_positive += 1
    elif predictions_test[i] == target_test[i] == 0:
        correct_negative += 1
    else:
        wrong +=1

for i in range(len(data_test)):
    label = target_test[i]
    class_total[label] += 1

negative_test_acc = correct_negative/class_total[0]
positive_test_acc = correct_positive/class_total[1]

print('Test Loss: {:.4f} | '.format(loss_test_avg) + ' Test Acc: {:.4f} | '.format(acc_test))
print('Positive test Acc: {:.4f} | '.format(positive_test_acc) + ' Negative test Acc: {:.4f} | '.format(negative_test_acc))

precision_recall_fscore_support(target_test, predictions_test, average='weighted')