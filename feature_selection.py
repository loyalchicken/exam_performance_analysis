from model import regression
import numpy as np
import pandas as pd 

def powerset(nums):
    def backtrack(first = 0, curr = []):
        if len(curr) == k:  
            output.append(curr[:])
        for i in range(first, n):
            curr.append(nums[i])
            backtrack(i + 1, curr)
            curr.pop()
            
    output = []
    n = len(nums)
    for k in range(n + 1):
        if k!=0:
            backtrack()
    return output

def subset_selection(best_model, train_data, val_data, test_data, train_labels, val_labels, test_labels):
    sets = powerset(np.arange(6))
    losses = []
    for i in range(len(sets)):
        train_loss, val_loss, test_loss = regression(best_model["Model"], best_model['Hyperparameters'][0], best_model['Hyperparameters'][1], best_model['Hyperparameters'][2], train_data[sets[i]], val_data[sets[i]], test_data[sets[i]], train_labels, val_labels, test_labels)
        losses.append([sets[i], train_loss, val_loss, test_loss])
    losses = pd.DataFrame.from_records(losses).rename(columns = {0:"Features", 1: "Training Loss", 2:"Validation Loss", 3:"Test Loss"}).set_index("Features")
    return losses.sort_values(by = "Validation Loss", ascending = True)