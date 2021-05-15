import gzip
import numpy as np
import pandas as pd
import math
import pyswarm as ps
import GWO
import random

def import_file(file):
    data_file = gzip.open(file)
    data = np.load(data_file)
    return data

def determine_true_rankscore(x):
    if x['booking_bool'] == 1:
        return 5
    elif x['click_bool'] == 1:
        return 1
    else:
        return 0

def prep_true_labels(true_labels):
    true_labels['true_label'] = true_labels.apply(lambda x: determine_true_rankscore(x), axis=1)
    true_labels = true_labels.drop(columns=['click_bool', 'booking_bool'], axis=1)
    return true_labels

def dcg_prediction(temp_pred, temp_true):
    '''
    Calculate CDG of predicted rank
    '''
    temp_true = temp_true.drop(columns=['search_id'], axis=1)
    merge = temp_pred.merge(temp_true, on=['hotel_id'])
    pred = merge['true_label'][:5].values
    return pred[0] + (pred[1]/math.log(2,2)) + (pred[2]/math.log(3,2)) + (pred[3]/math.log(4,2))

def dcg_ideal(temp_true):
    '''
    Calculate CDG of ideal ranking
    '''
    ideal = temp_true.sort_values(by=['true_label'], ascending=False)
    ideal = ideal['true_label'][:5].values
    return ideal[0] + (ideal[1]/math.log(2,2)) + (ideal[2]/math.log(3,2)) + (ideal[3]/math.log(4,2))


def evaluate_result(result, true_labels):
    '''
    Evaluate predicted rank based on true labels, using NDCG.
    '''
    results = []
    
    for search_id in result['search_id'].unique():
        temp_true = true_labels[true_labels['search_id']==search_id]
        idcg = dcg_ideal(temp_true)
        
        temp_pred = result[result['search_id']==search_id]
        dcg = dcg_prediction(temp_pred, temp_true)
        
        ndcg = dcg/idcg
        results.append(ndcg)

    return sum(results)/len(results)
    

def rank_output(data):
    '''
    Rank hotel_id's per search_id based on the rankscore.
    '''
    res_search_ids = []
    res_hotel_ids = []
    
    for search_id in data['search_id'].unique():
        temp = data[data['search_id']==search_id]
        temp = temp.sort_values(by=['rankscore'], ascending=False)
        res_search_ids.append(temp['search_id'].values)
        res_hotel_ids.append(temp['hotel_id'].values)
        
    res_search_ids = np.concatenate(res_search_ids)
    res_hotel_ids = np.concatenate(res_hotel_ids)
        
    result = pd.DataFrame(columns=('search_id', 'hotel_id'))
    result['search_id'] = res_search_ids
    result['hotel_id'] = res_hotel_ids
    return result
    

def rank_score(data, weights):
    '''
    Calculate rankscore for hotel_id to determine rank
    '''
    return weights[0]*data['prob0'] + weights[1]*data['prob1'] + weights[2]*data['prob5']

def init_weights():
    '''
    Initialize weights for rankscore function
    '''
    weights = [0.1, 1.0, 5.0]
    return weights

def fitness(weights, data, true_labels):
    data['rankscore'] = rank_score(data, weights)
    result = rank_output(data)
    ncdg = evaluate_result(result, true_labels)
    return ncdg
    
def grey_wolf_optimization(fitness_function, data, true_labels):
    print("Begin grey wolf optimization")
    dim = 3
    minW = -10.0
    maxW = 10.0
    
    print("Goal is to optimize weights w1, w2, w3 that determine the rankscore")
    
    num_particles = 50
    max_iter = 30
    
    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter    = " + str(max_iter))
    print("Lower bound weights = " + str(minW))
    print("Upper bound weights = " + str(maxW))
    print("\nStarting GWO algorithm\n")
    
    best_position = GWO.gwo(fitness_function, max_iter, num_particles, dim, minW, maxW, data, true_labels)
    
    print("\nGWO completed\n")
    print("\nBest solution found:")
    print(["%.6f"%best_position[k] for k in range(dim)])
    err = fitness(best_position, data, true_labels)
    print("fitness of best solution = %.6f" % err)
      
    print("\nEnd GWO for rastrigin\n")
    return


def main():
    path = "C:/Users/doist/OneDrive/Documenten/Business Analytics/Master/Year 1/Data Mining Techniques/Assignment 2/Data/"
    data = pd.read_csv(path+'simple_example.csv', sep=',')
    
    true_labels = pd.read_csv(path+'simple_example_true_labels.csv')
    true_labels = prep_true_labels(true_labels)
    
    weights = init_weights()
    data['rankscore'] = rank_score(data, weights)
    
    result = rank_output(data)
    
    ndcg = evaluate_result(result, true_labels)
    print(ndcg)
    
    grey_wolf_optimization(fitness, data, true_labels)

    

if __name__ == '__main__':
    main()