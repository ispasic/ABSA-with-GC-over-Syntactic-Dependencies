import numpy as np
import pickle
import torch

def data_setup(data, edge_type):

    indict = open('glove.840B.300d_dict.pickle', 'rb')
    dictionary = pickle.load(indict)

    embedding_size = 300
    
    if edge_type == 'original':
        
        dependency_tree = data.dependency_tree_clean_norm
        
        edges_list_tensors = []
        for i in dependency_tree:
            edges_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))
        
        sosy_idx = data.aspect_idx_norm
        idx_list_tensors = []
        for i in sosy_idx:
            idx_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))    
    
        sentiment_labels = data.sentiment    
        target_list = []
        for i in sentiment_labels:
            if i == -1:
                target_list.append(0)
            elif i == 1:
                target_list.append(1)
            else:
                print('Sentiment labels -1 or 1.')

        target = torch.tensor(target_list, dtype=torch.long)
    
        tokens = data.tokens_norm
        vertices_list_tensors = []
        for t in tokens:

            matrix_len = len(t)
            weights_matrix = np.zeros((matrix_len, embedding_size))
            words_found = 0

            for i, word in enumerate(t):
                try: 
                    weights_matrix[i] = dictionary[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = weights_matrix[i]
    
            vertices_list_tensors.append(torch.as_tensor(weights_matrix, dtype=torch.float))
            print("Loading {}/{} words from vocab.".format(words_found, len(t)))
  
         
    
    elif edge_type == 'undirected':
        
        dependency_tree = data.dependency_tree_clean_norm.to_list()
        dependency = data.dependency_norm.to_list()
        
        for i in range(len(dependency_tree)):
            for j in range(len(dependency_tree[i])):
                dependency_tree[i].append([dependency_tree[i][j][1],dependency_tree[i][j][0]])
                dependency[i].append(dependency[i][j])
    
    
        edges_list_tensors = []
        for i in dependency_tree:
            edges_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))
        
        sosy_idx = data.aspect_idx_norm
        idx_list_tensors = []
        for i in sosy_idx:
            idx_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))    
    
        sentiment_labels = data.sentiment    
        target_list = []
        for i in sentiment_labels:
            if i == -1:
                target_list.append(0)
            elif i == 1:
                target_list.append(1)
            else:
                print('Sentiment labels -1 or 1.')

        target = torch.tensor(target_list, dtype=torch.long)
    
        tokens = data.tokens_norm
        vertices_list_tensors = []
        for t in tokens:

            matrix_len = len(t)
            weights_matrix = np.zeros((matrix_len, embedding_size))
            words_found = 0

            for i, word in enumerate(t):
                try: 
                    weights_matrix[i] = dictionary[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = weights_matrix[i]
    
            vertices_list_tensors.append(torch.as_tensor(weights_matrix, dtype=torch.float))
            print("Loading {}/{} words from vocab.".format(words_found, len(t)))
            
            

    elif edge_type == 'undirected_neg':
        
        dependency_tree = data.dependency_tree_clean.to_list()
        dependency = data.dependency.to_list() 
        
        for i in range(len(dependency)):
            for j in range(len(dependency[i])):
                if dependency[i][j] == 'neg':
                    dependency[i].append(dependency[i][j])
                    dependency_tree[i].append([dependency_tree[i][j][1], dependency_tree[i][j][0]])
            
        edges_list_tensors = []
        for i in dependency_tree:
            edges_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))
        
        sosy_idx = data.aspect_idx_norm
        idx_list_tensors = []
        for i in sosy_idx:
            idx_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))    
    
        sentiment_labels = data.sentiment    
        target_list = []
        for i in sentiment_labels:
            if i == -1:
                target_list.append(0)
            elif i == 1:
                target_list.append(1)
            else:
                print('Sentiment labels -1 or 1.')

        target = torch.tensor(target_list, dtype=torch.long)
    
        tokens = data.tokens
        vertices_list_tensors = []
        for t in tokens:

            matrix_len = len(t)
            weights_matrix = np.zeros((matrix_len, embedding_size))
            words_found = 0

            for i, word in enumerate(t):
                try: 
                    weights_matrix[i] = dictionary[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = weights_matrix[i]
    
            vertices_list_tensors.append(torch.as_tensor(weights_matrix, dtype=torch.float))
            print("Loading {}/{} words from vocab.".format(words_found, len(t)))
            
    
           
    elif edge_type == 'reverse_neg':
        
        dependency_tree_orig = data.dependency_tree_clean.to_list()
        dependency_orig = data.dependency.to_list() 
        
        dependency = [[] for i in range(len(dependency_orig))]
        dependency_tree = [[] for i in range(len(dependency_orig))]
        
        
        for i in range(len(dependency_orig)):
            for j in range(len(dependency_orig[i])):
                if dependency_orig[i][j] != 'neg':
                    dependency[i].append(dependency_orig[i][j])
                    dependency_tree[i].append(dependency_tree_orig[i][j])
                else:
                    dependency[i].append(dependency_orig[i][j])
                    dependency_tree[i].append([dependency_tree_orig[i][j][1], dependency_tree_orig[i][j][0]])

        edges_list_tensors = []
        for i in dependency_tree:
            edges_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))
        
        sosy_idx = data.aspect_idx_norm
        idx_list_tensors = []
        for i in sosy_idx:
            idx_list_tensors.append(torch.as_tensor(i, dtype=torch.int64))    
    
        sentiment_labels = data.sentiment    
        target_list = []
        for i in sentiment_labels:
            if i == -1:
                target_list.append(0)
            elif i == 1:
                target_list.append(1)
            else:
                print('Sentiment labels -1 or 1.')

        target = torch.tensor(target_list, dtype=torch.long)
    
        tokens = data.tokens
        vertices_list_tensors = []
        for t in tokens:

            matrix_len = len(t)
            weights_matrix = np.zeros((matrix_len, embedding_size))
            words_found = 0

            for i, word in enumerate(t):
                try: 
                    weights_matrix[i] = dictionary[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = weights_matrix[i]
    
            vertices_list_tensors.append(torch.as_tensor(weights_matrix, dtype=torch.float))
            print("Loading {}/{} words from vocab.".format(words_found, len(t)))
            
     
        
    else:
        print('Wrong input argument. Edge type: original/undirected/undirected_neg/reverse_neg')        
        
        
    return edges_list_tensors, vertices_list_tensors, idx_list_tensors, target