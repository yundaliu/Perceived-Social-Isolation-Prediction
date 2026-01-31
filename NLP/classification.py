import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import re
import argparse
import numpy as np
import yaml
from collections import defaultdict

from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import SCORERS
from sklearn.pipeline import Pipeline

import torch

from script.directory import social_isolation_v2_dir
from script.dataset import generate_embeddings, generate_filename
from script import network
import script.sklearn_train as script_train
import script.subjects as script_subjects

# input_dir = os.path.join(
#     social_isolation_v2_dir,
#     'input'
# )

workspace = os.path.join(
    '/scratch3',
    'workspace',
    'yundaliu_umass_edu-simple'
)

input_dir = os.path.join(
    workspace,
    'social_isolation_input'
)

output_dir = os.path.join(
    social_isolation_v2_dir,
    'out_classification_v2',
    'data'
)

config_dir = os.path.join(
    social_isolation_v2_dir,
    'config'
)


def main():
    ########################################
    # Load parameters
    ########################################
    args = parse_arguments()
    print(f'TASK_ID:                   {args.task_id}', flush=True)
    print(f'PREV:                      {args.prev}', flush=True)
    print(f'NETWORK_DIR:               {args.network_dir}', flush=True)
    print(f'MODEL_CONFIG_NUMBER:       {args.model_config_number}', flush=True)
    print('\n', flush=True)
    
    ########################################
    # Load model config
    ########################################
    # Model config file
    args.model_config = load_config(
        os.path.join(
            config_dir,
            'Model',
            'Model_{:02d}.yaml'.format(args.model_config_number))
    )
    print('Model Config', flush=True)
    print_config(args.model_config)
    
    ########################################
    # Parse network_dir
    ########################################
    network_dir = args.network_dir.split(',')

    ########################################
    # Load data
    ########################################
    print('Loading data...', flush=True)
    match = re.search(r'Data_(\d+)', network_dir[0])
    args.data_config_number = int(match.group(1))
    args.data_config = load_config(
        os.path.join(
            config_dir,
            'Data',
            'Data_{:02d}.yaml'.format(args.data_config_number))
    )
    print('Data Config', flush=True)
    print_config(args.data_config)
    
    filename = generate_filename(args)
    if filename not in os.listdir(input_dir):
        if args.task_id != 1:
            exit()
        generate_embeddings(args)

    data = np.load(os.path.join(input_dir, filename), allow_pickle=True)
    embeddings = data['features']
    feature_names = data['feature_names']
    labels = data['labels']
    subjects = data['subjects']
    n_subjects = len(subjects)
   
    # Process subjects
    subjects_copy = np.zeros_like(subjects)
    for _, subj in enumerate(np.unique(subjects)):
        mask = subjects == subj
        subjects_copy[mask] = script_subjects.subject_dict[subj]
        del mask

    ########################################
    # Load model -- Auto-Encoder v1
    ########################################
    if len(network_dir) >= 1 and network_dir[0].endswith('v1'):
        print('Loading the first pretrained network...', flush=True)
        model = load_model(network_dir[0], args, embeddings)

        ########################################
        # Extract embeddings -- Auto-Encoder v1
        ########################################
        print('Extracting embeddings...', flush=True)
        device = check_cuda()
        embeddings = apply_model(
            embeddings,
            model,
            device
        )
        del model

        # embeddings shape: (n_subjects * args.prev, em_dim)
        # reshape embeddings
        n_surveys, em_dim = embeddings.shape
        assert n_surveys == n_subjects * args.prev, 'n_surveys != n_subjects * args.prev'
        embeddings = embeddings.reshape(n_subjects, args.prev, em_dim)
        print('embedding shape: ', embeddings.shape, flush=True)
        
        network_dir.pop(0)
    
    ########################################
    # Load model -- Auto-Encoder v2
    ########################################
    if len(network_dir) >= 1 and network_dir[0].endswith('v2'):
        n_samples, _, em_dim = embeddings.shape
        if n_samples == n_subjects * args.prev:
            embeddings = embeddings.reshape(n_subjects, args.prev, em_dim)
        
        print('Loading the second pretrained network...', flush=True)
        model = load_model(network_dir[0], args, embeddings)
    
        ########################################
        # Extract embeddings -- Auto-Encoder v2
        ########################################
        print('Extracting embeddings...', flush=True)
        device = check_cuda()
        embeddings = apply_model(
            embeddings,
            model,
            device
        )
        del model
    else:
        # average embeddings
        n_samples, _, em_dim = embeddings.shape
        if n_samples == n_subjects * args.prev:
            embeddings = embeddings.reshape(n_subjects, args.prev, em_dim)
        embeddings = np.mean(embeddings, axis=1)
    print('embedding shape: ', embeddings.shape, flush=True)

    ########################################
    # Classification
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

    X_train, X_test = embeddings[train_mask], embeddings[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]
    group_train, group_test = subjects_copy[train_mask], subjects_copy[test_mask]
    
    # Normalization
    scaler = StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Feature Selection
    if args.model_config['feature_selection'] is not None:
        fsel = script_train.feature_selection(
            X_train, y_train, group_train, feature_names, args)
        X_train = fsel.transform(X_train)
        X_test = fsel.transform(X_test)
    else:
        fsel = None
        
    # Gridsearch and train the model
    grid_search, clf = script_train.train(
        X_train, y_train, group_train, args)
    
    # Prediction
    test_labels[test_mask] = y_test
    predictions[test_mask] = clf.predict(X_test)
    if 'predict_proba' in dir(clf):
        scores[test_mask] = clf.predict_proba(X_test)
    else:
        d = clf.decision_function(X_test)
        test_score = np.zeros((len(d), 2))
        test_score[:, 1] = d
        scores[test_mask] = test_score
    test_subjects[test_mask] = group_test
    
    try:
        scorer = script_train.get_scorer(args.model_config['scoring'])
        results[args.task_id]['train_score'] = scorer._score_func(y_train, clf.predict(X_train), **scorer._kwargs)
        results[args.task_id]['cv_score'] = grid_search.best_score_
        results[args.task_id]['best_params'] = grid_search.best_params_
        if fsel is not None:
            results[args.task_id]['features'] = fsel.feature_to_select
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
    folder = 'prev_{:d}_{:s}_model_config_{:02d}'.format(
        args.prev, args.network_dir, args.model_config_number)
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
        type=int
    )
    parser.add_argument(
        '--network_dir',
        required=True,
        type=str
    )
    parser.add_argument(
        '--model_config_number',
        required=True,
        type=int
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as stream:
        return yaml.safe_load(stream)
    
    
def print_config(config_dict):
    for key, val in config_dict.items():
        print(key, '\t', val, flush=True)
    print('\n', flush=True)
    
    
def check_cuda(verbose=True):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        if verbose:
            print(f'CUDA VERSION: {torch.backends.cudnn.version()}')
            print(f'Using Device: {device}')
        return device
    else:
        raise EnvironmentError('Could not access GPU.')
        

def apply_model(features, model, device, batch_size=32):
    n_samples = features.shape[0]
    embeddings = []
    
    model.eval()
    model.to(device=device)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = features[i:i+batch_size]
            batch = torch.from_numpy(batch).to(device=device, dtype=torch.float)
            
            output = model(batch, downstream=True)
            
            embeddings.append(output.cpu().detach().numpy())
    return np.vstack(embeddings)


def load_model(network_dir, args, features):
    # Load network config
    match = re.search(r'Network_(\d+)', network_dir)
    network_config_number = int(match.group(1))
    network_config = load_config(
        os.path.join(
            config_dir,
            'Network',
            'Network_{:02d}.yaml'.format(network_config_number))
    )
    print('Network Config', flush=True)
    print_config(network_config)
    
    # Define model
    input_shape = features.shape
    model = network.init_model(
        network_config,
        input_shape
    )
    
    # Load pre-trained parameters
    model_dir = os.path.join(
        workspace,
        network_dir,
        '{:02d}.pt'.format(args.task_id)
    )
    saved_model = torch.load(model_dir, weights_only=True)
    model.load_state_dict(saved_model)
    return model

        
if __name__ == '__main__':
    main()