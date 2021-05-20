import numpy as np
import pandas as pd
import math
import GWO
import PSO
import pipeline_connector
from tqdm import tqdm
import random
import pickle

def determine_true_rankscore(x):
    if x['booking_bool'] == 1:
        return 5
    elif x['click_bool'] == 1:
        return 1
    else:
        return 0

def prep_true_labels(true_labels):
    '''
    Function to transform 'click_bool' and 'booking_bool' into 1 column with 0, 1 or 5
    '''
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
    return 0.0001 + ideal[0] + (ideal[1]/math.log(2,2)) + (ideal[2]/math.log(3,2)) + (ideal[3]/math.log(4,2))


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

    return -sum(results)/len(results)
    

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

def init_weights(equal=False):
    '''
    Initialize weights for rankscore function
    '''
    if equal:
        weights = [0.0, 0.0, 0.0]
    else:
        weights = [0.0, 1.0, 5.0]
    return weights

def fitness(weights, data, true_labels):
    data['rankscore'] = rank_score(data, weights)
    result = rank_output(data)
    ncdg = evaluate_result(result, true_labels)
    return ncdg
    
def grey_wolf_optimization(fitness_function, data, true_labels):
    print("Begin grey wolf optimization")
    dim = 3
    minW = -1.0
    maxW = 1.0
    
    print("Goal is to optimize weights w1, w2, w3 that determine the rankscore")
    
    num_particles = 10
    max_iter = 7
    
    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter    = " + str(max_iter))
    print("Lower bound weights = " + str(minW))
    print("Upper bound weights = " + str(maxW))
    print("\nStarting GWO algorithm\n")
    
    best_sol, gwo_summary = GWO.gwo(fitness_function, max_iter, num_particles, dim, minW, maxW, data, true_labels)
    f = open('gwo_summary2.pkl', 'wb')
    pickle.dump(gwo_summary,f)
    f.close()
    
    for i in range(max_iter):
        print('\nIteration ' + str(i))
        print('\nBest set of weights: ', gwo_summary[i]['best_solution'])
        print('\nFitness best solution: ', gwo_summary[i]['fitness_best_solution'])
    
    print("\nGWO completed\n")
    print("\nBest solution found:")
    print(["%.6f"%best_sol.weights[k] for k in range(dim)])
    err = fitness(best_sol.weights, data, true_labels)
    print("\nfitness of best solution = %.6f" % err)
      
    print("\nEnd GWO\n")
    return best_sol


def main():
    path = 'C:/Users/doist/OneDrive/Documenten/GitHub/DataMiningTechniques/Assignment2/Final Code/'
    data, true_labels = pipeline_connector.prep_data(path, '/HistGradientBoostingClassifier_top8_downsample_cv_proba.npy.gz')

    x_data = data[:1000]
    y_true_labels = true_labels[:1000]

    weights = init_weights(equal=False)
    
    # PIPELINE rank once:
    '''
    print('\nRanking with arbitrary weights: ', weights)
    data['rankscore'] = rank_score(data, weights)
    result = rank_output(data)
    ndcg = evaluate_result(result, true_labels)
    print('\nObtained NCDG = ', ndcg)
    '''
    
    # GWO
    data, true_labels = pipeline_connector.prep_data(path, '/HistGradientBoostingClassifier_top8_downsample_cv_proba.npy.gz')
    grey_wolf_optimization(fitness, data, true_labels)
    
    
    # PSO
    data, true_labels = pipeline_connector.prep_data(path, '/HistGradientBoostingClassifier_top8_downsample_cv_proba.npy.gz')
    weights = init_weights(equal=True)
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    num_particles = 10
    maxiter = 7
    print('PSO initialized with weights: ', weights)
    pso_summary = PSO.PSO.particle_swarm_optimization(fitness, weights, bounds, num_particles, maxiter, data, true_labels)
    f = open('pso_summary2.pkl', 'wb')
    pickle.dump(pso_summary,f)
    f.close()
    
    
    for i in range(maxiter):
        print('\nIteration ' + str(i))
        print('Best set of weights: ', pso_summary[i]['best_solution'])
        print('Fitness best solution: ', pso_summary[i]['fitness_best_solution'])
        
        #for part in summary[i]['swarm']:
         #   print(part.weights_i)
         #   print(part.ndcg_i)
    return
    

if __name__ == '__main__':
    main()