import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import yaml

from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import SCORERS
from sklearn.pipeline import Pipeline

import script
import script.features as script_features
import script.train as script_train
import script.subjects as script_subjects
from nn_utils.extract_text_embeddings import convert_text_to_embeddings

import warnings
warnings.filterwarnings('ignore')

home_dir = os.path.join(
    '/home',
    'yundaliu_umass_edu',
    'social_isolation'
)

input_dir = os.path.join(
    home_dir,
    'input'
)

output_dir = os.path.join(
    home_dir,
    'output_v2',
    'data'
)

config_dir = os.path.join(
    home_dir,
    'config',
)

def main():
    ########################################
    # Load parameters
    ########################################
    args = parse_arguments()
    print(f'TASK_ID:                   {args.task_id}', flush=True)
    print(f'PREV:                      {args.prev}', flush=True)
    print(f'FEATURE_CONFIG_NUMBER:     {args.feature_config_number}', flush=True)
    print(f'GRID_SEARCH_SCORING:       {args.grid_search_scoring}', flush=True)
    print(f'MODEL:                     {args.model}', flush=True)
    print(f'MODEL_CONFIG_NUMBER:       {args.model_config_number}', flush=True)

    ########################################
    # Load feature_config
    ########################################
    args.feature_config = load_config(
        os.path.join(
            config_dir,
            'feature',
            'Feature_{:02d}.yaml'.format(args.feature_config_number))
    )
    print('FEATURE CONFIG', flush=True)
    print_config(args.feature_config)
    
    ########################################
    # Load model_config
    ########################################
    args.model_config = load_config(
        os.path.join(
            config_dir,
            'model',
            '{:s}_{:02d}.yaml'.format(args.model, args.model_config_number))
    )
    print('MODEL CONFIG', flush=True)
    print_config(args.model_config)

    ########################################
    # Load input data
    ########################################
    npz_data = np.load(os.path.join(input_dir, 'survey_data.npz'), allow_pickle=True)
    survey_data = npz_data['data'].item()
    del npz_data
    
    npz_data = np.load(os.path.join(input_dir, 'demographics.npz'), allow_pickle=True)
    demographics = npz_data['data'].item()
    del npz_data
    
    ema_survey2_cols = pd.read_csv(
        os.path.join(input_dir, 'ema_survey2_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()
    
    demographics_cols = pd.read_csv(
        os.path.join(input_dir, 'demographics_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()
    
    ########################################
    # Extract features
    ########################################
    print('Extracting features...', flush=True)
    if args.feature_config['use_embeddings']:
        emFilename = 'prev_{:02d}_embeddings_{:02d}.npz'.format(
            args.prev, args.feature_config['use_embeddings'])
        emData = np.load(os.path.join(input_dir, 'Embeddings', emFilename), allow_pickle=True)
        features = emData['features']
        feature_names = emData['feature_names']
        labels = emData['labels']
        subjects = emData['subjects']
        del emData, emFilename
    else:
        features, feature_names, labels, subjects, feature_keys, label_keys = script_features.extract_features(
            survey_data, ema_survey2_cols, demographics, demographics_cols, args)
        
    # Process subjects
    subjects_copy = np.zeros_like(subjects)
    for _, subj in enumerate(np.unique(subjects)):
        mask = subjects == subj
        subjects_copy[mask] = script_subjects.subject_dict[subj]
        del mask
    
    features, feature_names = script_features.process_features(
        features, feature_names, args)
    
    # Remove invalid features
    features, labels, feature_names, subjects = script_features.remove_invalid_features(
        features, labels, feature_names, subjects)
    
    n_samples, n_features = features.shape
    print('{:d} samples.'.format(n_samples), flush=True)
    print('{:d} features.'.format(n_features), flush=True)
    
    ########################################
    # Train the model
    ########################################
    # Convert labels to binary classes
    labels_mask = (labels == 1)
    labels[labels_mask] = 0
    labels[~labels_mask] = 1
    n_labels = len(np.unique(labels))
    print(np.unique(labels, return_counts=True), flush=True)
    
    # Save variables
    n_samples = len(labels)
    test_labels = np.zeros((n_samples, ))
    test_labels[:] = np.nan
    predictions = test_labels.copy()
    scores = np.zeros((len(labels), n_labels))
    scores[:] = np.nan
    test_subjects = test_labels.copy()
    
    results = defaultdict(dict)
    models = defaultdict(dict)
        
    # Train
    print('Training...', flush=True)
    print('Test on subject {:d} - {:d}'.format((args.task_id - 1) * 10, args.task_id * 10 - 1), flush=True)
    test_subject = np.arange((args.task_id - 1) * 10, args.task_id * 10, dtype=int)
    
    # Split Train/Test Set
    test_mask = np.isin(subjects_copy, test_subject)
    train_mask = ~test_mask

    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]
    group_train, group_test = subjects_copy[train_mask], subjects_copy[test_mask]
    
    # Normalization
    scaler = StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature selection
    if args.feature_config['feature_selection'] is not None:
        fsel = script_train.feature_selection(X_train, y_train, group_train, feature_names, args)
        X_train = fsel.transform(X_train)
        X_test = fsel.transform(X_test)
    else:
        fsel = None

    # Gridsearch and train the model
    grid_search, clf = script_train.train(X_train, y_train, group_train, args)

    # Prediction
    test_labels[test_mask] = y_test
    predictions[test_mask] = clf.predict(X_test)
    if 'predict_proba' in dir(clf):
        scores[test_mask] = clf.predict_proba(X_test)
    else:
        d = clf.decision_function(X_test)
        prob = np.exp(d) / np.sum(np.exp(d))
        scores[test_mask][:, 0] = 1 - prob
        scores[test_mask][:, 1] = prob
    test_subjects[test_mask] = group_test

    try:
        scorer = script_train.get_scorer(args)
        results[args.task_id]['train_score'] = scorer._score_func(y_train, clf.predict(X_train), **scorer._kwargs)
        results[args.task_id]['cv_score'] = grid_search.best_score_
        results[args.task_id]['best_params'] = grid_search.best_params_
        if fsel is not None:
            results[args.task_id]['features'] = fsel.feature_to_select

            if args.feature_config['feature_selection'] == 'AUROCSequentialFeatureSelector':
                results[args.task_id]['support'] = fsel.fsel.support_
        elif isinstance(clf, Pipeline):
            results[args.task_id]['features'] = clf.named_steps['feature_selection'].feature_to_select

        if isinstance(clf, Pipeline):
            models[args.task_id] = clf.named_steps['classifier']
        else:
            models[args.task_id] = clf
        print('Score on training set: {:.2f}'.format(results[args.task_id]['train_score']), end='\t', flush=True)
        print('Score on test set: {:.2f}'.format(scorer._score_func(y_test, predictions[test_mask], **scorer._kwargs)), flush=True)
    except Exception as e:
        print(e)
    
    ########################################
    # Save
    ########################################
    folder = '{:s}_prev_{:d}_feature_config_{:d}_grid_search_scoring_{:s}_model_config_{:02d}'.format(
        args.model, args.prev, args.feature_config_number, 
        args.grid_search_scoring, args.model_config_number)
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    output_data = {
        'y_test': test_labels,
        'y_test_pred': predictions,
        'y_test_score': scores,
        'test_subjects': test_subjects,
        'results': results,
        'models': models
    }
    filename = 'task_id_{:03d}.npz'.format(args.task_id)
    np.savez(os.path.join(output_dir, folder, filename), **output_data)
    print('DONE.', flush=True)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_id',
        required=True,
        type=int
    )
    parser.add_argument(
        '--prev',
        required=True,
        type=int,
        default=1
    )
    parser.add_argument(
        '--feature_config_number',
        required=True,
        type=int,
        default=1
    )
    parser.add_argument(
        '--grid_search_scoring',
        required=True,
        type=str,
        default='roc_auc'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='RandomForestClassifier',
        choices=['RandomForestClassifier', 
                 'LogisticRegression', 
                 'XGBoost', 
                 'LGBMClassifier',
                 'RidgeClassifier',
                 'SGDClassifier',
                 'GPBoostClassifier',
                 'CatBoostClassifier',
                 'AUPRCThreshold+LogisticRegression']
    )
    parser.add_argument(
        '--model_config_number',
        required=True,
        type=int,
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as stream:
        return yaml.safe_load(stream)
    
    
def print_config(config_dict):
    for key, val in config_dict.items():
        print(key, '\t', val, flush=True)
    print('\n', flush=True)

    
if __name__ == '__main__':
    main()
    