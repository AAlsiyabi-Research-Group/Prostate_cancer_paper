"""
Genetic Algorithm–Based Feature Selection (Parallel Implementation)
-------------------------------------------------------------------
Authors: Syed Ahsan Shahid, Ahmed Al-Harrasi, Adil Al-Siyabi
Affiliation: Natural and Medical Sciences Research Center, University of Nizwa, Oman
Year: 2025

Paper:
"A Minimal Plasma Proteome-Based Biomarker Panel for Accurate Prostate Cancer Diagnosis"
(Syed Ahsan Shahid, Ahmed Al-Harrasi, Adil Al-Siyabi, 2025)

Dataset:
Compatible with the **Olink Explore 1536 pan-cancer plasma proteomics dataset**
published by Álvez et al. (2023) in *Nature Communications*.
Dataset access link:
https://www.nature.com/articles/s41467-023-39765-y

Note:
This repository does **not redistribute** the dataset; users can directly
download it from the above publication or the Olink data portal.
The provided script demonstrates the GA-based feature-selection framework
used in the study and can be applied to any similar tabular or omics dataset.

"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import math

from sklearn.utils import shuffle
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.callbacks import ConsecutiveStopping

# Replace this with your own dataset path
data = pd.read_excel('path_to_dataset.xlsx', index_col=0) # Note: Dataset not included here.


# Define cancer types
cancer_types = ['PRC', 'AML', 'BRC', 'CLL', 'CRC', 'CVX', 'ENDC', 'GLIOM', 'LUNGC', 'LYMPH', 'MYEL', 'OVC']

# Create output folder for results
io_folder = 'cancer_data_results'
if not os.path.exists(io_folder):
    os.makedirs(io_folder)

def prepare_data(cancer_type):
    """Prepare data for the given cancer type."""
    # Encode the cancer type as binary
    data['CancerTypeBinary'] = (data['Cancer'] == cancer_type).astype(int)
    # Gather cases for the cancer type
    cancer_cases = data[data['CancerTypeBinary'] == 1]
    # Sample an equal number of non-cancer cases for balance
    non_cancer_cases = data[data['CancerTypeBinary'] == 0].sample(n=len(cancer_cases), random_state=42)
    # Combine and shuffle to create a balanced dataset
    balanced_data = pd.concat([cancer_cases, non_cancer_cases])
    balanced_data = shuffle(balanced_data, random_state=42)
    # Extract features and labels
    X = balanced_data.drop(['Cancer', 'CancerTypeBinary'], axis=1)
    y = balanced_data['CancerTypeBinary']
    return X, y

ab = 'PRC'  # Specify the cancer type to process
X, y = prepare_data(ab)  # Prepare data for the specified cancer type

print(f'Processed data for {ab}:')
print(f'Features (X) shape: {X.shape}')
print(f'Labels (y) shape: {y.shape}')

# Parameters for genetic algorithm
iterations = 1
feature_sample = [40]
scoring_list = ['roc_auc']
ab = 'YourCancerType'  # e.g., 'PRC'
algo = 'LR'
ratio = '91'
processors = 8 #authors used 40 processes
processes = 50 #authors used 1000 processes
pop_size=1000
gen_count=300


cross_probability = float('0.'+ratio[0])
mut_probability = float('0.'+ratio[1])


def run_generation(parameter):
    
    gen_start_time = time.time()
    print(f'X: {X.shape}, y: {y.shape}')
    X_columns = X.columns
  
    callback = ConsecutiveStopping(generations=10, metric='fitness_max')

    file_name_counter = parameter

    for metric in scoring_list:
        #write to an excel file
        path = str(os.getcwd())+f'/{io_folder}/cancer_scoring-{metric}_{file_name_counter}.xlsx'
        writer = pd.ExcelWriter(path, engine = 'openpyxl')
                      
        for item in feature_sample:
            
            if algo == 'SVM':
                clf = svm.SVC()
            elif algo == 'LR':
                clf = LogisticRegression()
                
            top_iteration_df = pd.DataFrame()
            top_iteration_dict = []
            gene_count = {key: 0 for key in X_columns}

            for iteration in range(iterations):
                print(f"===== AB: {ab} *** Algo: {algo} *** Scoring: {metric} *** Max Feature Size: {item} *** Iteration: {iteration+1} *** Ratio: {ratio} =====")
                evolved_estimator = GAFeatureSelectionCV(
                    estimator=clf,
                    cv=4,
                    scoring=metric,
                    population_size=pop_size,
                    max_features=item,
                    generations=gen_count,
                    verbose=True,
                    keep_top_k=2,
                    elitism=True,
                    crossover_probability=cross_probability,
                    mutation_probability=mut_probability,
                    criteria='max',
                )

                evolved_estimator.fit(X, y, callbacks=callback)
                features = evolved_estimator.support_
                best_features = evolved_estimator.best_features_
                feature_count = sum(best_features)
                feature_list = []
                for index,value in enumerate(best_features):
                    if value == True:
                        feature_list.append(X_columns[index])

                history_dict = evolved_estimator.history
                history_df = pd.DataFrame(history_dict)
                history_df = history_df.sort_values(by=['fitness_max'], ascending=False)

                ## implement the top iteration df
                top_iteration_row = history_df.iloc[0]
                top_iteration_row_dict = {'gen': int(top_iteration_row['gen']),'genes': str(feature_list), 'fitness': top_iteration_row['fitness'], \
                                          'fitness_std': top_iteration_row['fitness_std'], 'fitness_max': top_iteration_row['fitness_max'], \
                                          'fitness_min': top_iteration_row['fitness_min']}

                print(f'**************************{top_iteration_row_dict}******************************')

                feature_df = pd.DataFrame(feature_list)

                for feature in feature_list:
                    if feature in gene_count:
                        gene_count[feature] += 1
                gene_count_df = pd.DataFrame.from_dict(gene_count, orient='index', columns=['count'])
                gene_count_df = gene_count_df.sort_values(by=['count'], ascending=False)

            history_df.to_excel(writer, index=False, sheet_name=f'{item}-result')
            feature_df.to_excel(writer, index=False, sheet_name=f'{item}-best_features')

        writer.close()
        print(f"========= Generation Runtime: {round((time.time() - gen_start_time)/60,2)} minutes =========")


def run_parallel():

    # number of processes
    process_count = [x for x in range(processes)]
    
    processor_count = processors

    for iteration in range(math.ceil(len(process_count)/processor_count)):
        print(iteration*processor_count)
        small_process_list = [x for x in range(iteration*processor_count, iteration*processor_count+processor_count,1) if x < len(process_count)]
        pool = Pool(processes=processor_count)
        pool.map(run_generation, small_process_list)
    

def combine_result():

    file_list = os.listdir(io_folder)
    top_iteration_df = pd.DataFrame()
    top_iteration_dict = []
    gene_count = {key: 0 for key in X.columns}
    
    path = str(os.getcwd())+f'/{io_folder}/master#{ab}#({algo}-{feature_sample[0]}-{scoring_list[0]}-{ratio})#{processes}.xlsx'
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    
    for file in file_list:
        if os.path.isfile(f'{io_folder}/{file}'):
            if 'master' not in file:
                result_df = pd.read_excel(f'{io_folder}/{file}', sheet_name=0)
                feature_df = pd.read_excel(f'{io_folder}/{file}', sheet_name=1)

                #result df creation
                top_iteration_row = result_df.iloc[0:1]
                gene_names = feature_df[0].tolist()
                top_iteration_row_dict = {'gen': top_iteration_row['gen'].values[0],'genes': gene_names, 'gene_count': len(gene_names),\
                                          'fitness': top_iteration_row['fitness'].values[0], 'fitness_std': top_iteration_row['fitness_std'].values[0], \
                                          'fitness_max': top_iteration_row['fitness_max'].values[0], 'fitness_min': top_iteration_row['fitness_min'].values[0]}
                
                top_iteration_dict.append(top_iteration_row_dict)
                

                #feature count df creation
                for gene in feature_df.itertuples():
                    feature = gene[1]
                    if feature in gene_count:
                        gene_count[feature] += 1
                os.remove(f'{io_folder}/{file}')
                
    top_iteration_df = pd.DataFrame(top_iteration_dict)
    top_iteration_df = top_iteration_df.sort_values(by=['fitness_max'], ascending=False)
    gene_count_df = pd.DataFrame.from_dict(gene_count, orient='index', columns=['count'])
    gene_count_df = gene_count_df.sort_values(by=['count'], ascending=False)
    print("**********************************************")
    print(top_iteration_df.head)
    print(gene_count_df.head())
    print("**********************************************")
    top_iteration_df.to_excel(writer, index=False, sheet_name='top_iteration_rows')
    gene_count_df.to_excel(writer, index=True, sheet_name='gene_count')
    writer.close()

# Driver code
if __name__ == '__main__':
    start_time = time.time()
    run_parallel()
    combine_result()
    print(f"Runtime: {round((time.time() - start_time)/60,2)} minutes")
