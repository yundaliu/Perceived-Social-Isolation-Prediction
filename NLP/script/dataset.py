import os
import numpy as np
import pandas as pd
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader

from script.directory import social_isolation_dir, social_isolation_v2_dir
import script.features as script_features
from script.extract_text_embeddings import convert_text_to_embeddings
import script.subjects as script_subjects


work_dir = os.path.join(
    '/scratch3',
    'workspace',
    'yundaliu_umass_edu-simple'
)

# input_dir = os.path.join(
#     social_isolation_v2_dir,
#     'input'
# )

input_dir = os.path.join(
    work_dir,
    'social_isolation_input'
)

np.random.seed(0)
torch.manual_seed(42)


def generat_features():
    data_dir = os.path.join(
        social_isolation_dir,
        'input'
    )

    # Load input data
    npz_data = np.load(
        os.path.join(data_dir, 'survey_data.npz'), 
        allow_pickle=True)
    survey_data = npz_data['data'].item()
    del npz_data

    npz_data = np.load(
        os.path.join(data_dir, 'demographics.npz'), 
        allow_pickle=True)
    demographics = npz_data['data'].item()
    del npz_data

    ema_survey2_cols = pd.read_csv(
        os.path.join(data_dir, 'ema_survey2_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()

    demographics_cols = pd.read_csv(
        os.path.join(data_dir, 'demographics_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()
    
    print('Extracting features...', flush=True)
    args_copy = SimpleNamespace()
    args_copy.prev = 0
    args_copy.feature_config = {'use_embeddings': False}
    features, feature_names, labels, subjects, feature_keys, label_keys = script_features.extract_features(
        survey_data, ema_survey2_cols, demographics, demographics_cols, args_copy)
    return features, feature_names, labels, subjects, feature_keys, label_keys
    
    
def generate_embeddings(args):
    data_dir = os.path.join(
        social_isolation_dir,
        'input'
    )

    # Load input data
    npz_data = np.load(
        os.path.join(data_dir, 'survey_data.npz'), 
        allow_pickle=True)
    survey_data = npz_data['data'].item()
    del npz_data

    npz_data = np.load(
        os.path.join(data_dir, 'demographics.npz'), 
        allow_pickle=True)
    demographics = npz_data['data'].item()
    del npz_data

    ema_survey2_cols = pd.read_csv(
        os.path.join(data_dir, 'ema_survey2_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()

    demographics_cols = pd.read_csv(
        os.path.join(data_dir, 'demographics_cols.txt'),
        lineterminator = '\n', header = None
    )[0].to_list()

    # Setup
    args_copy = SimpleNamespace()
    args_copy.prev = args.prev
    args_copy.feature_config = {
        'use_embeddings': True,
        'use_demographics': args.data_config['use_demographics'],
    }
    if 'use_timestamp' in args.data_config:
        args_copy.feature_config['use_timestamp'] = args.data_config['use_timestamp']
    
    # Extract sentences
    print('Extracting features...', flush=True)
    features, feature_names, labels, subjects, feature_keys, label_keys = script_features.extract_features(
        survey_data, ema_survey2_cols, demographics, demographics_cols, args_copy)
    
    # Reshape data
    features = np.squeeze(features)
    if len(features.shape) == 2:
        features = features[:, None, :]
    print(features.shape, flush=True)
    del args_copy
        
    # Extract embeddings
    embeddings = convert_text_to_embeddings(features, args.data_config['embedding_config'])
    ## if prev == 0, features.shape is (n_surveys, n_questions, em_dim)
    ## if prev >= 1, features.shape is (n_surveys * prev, n_questions, em_dim)
    # Reshape features
    if args.data_config['embedding_config']['text_level'] == 'survey':
        embeddings = embeddings[:, None, :]
    print(embeddings.shape, flush=True)

    # Save embeddings
    output = {
        'features': embeddings,
        'feature_names': feature_names,
        'labels': labels,
        'subjects': subjects,
        'feature_keys': feature_keys,
        'label_keys': label_keys
    }
    
    # Save filename
    filename = generate_filename(args)
    if filename not in os.listdir(input_dir):
        np.savez(os.path.join(input_dir, filename), **output)
    return embeddings, feature_names, labels, subjects, feature_keys, label_keys


def generate_filename(args):
    base_filename = 'embeddings'
    filename = '{:s}_{:s}_data_config_{:d}_prev_{:d}.npz'.format(
        args.data_config['embedding_config']['pretrained_model_name'].replace('/', '_'),
        base_filename,
        args.data_config_number,
        args.prev
    )
    return filename
    
    
def dataset_factory(args, input_data=None):
    # Load embeddings
    if input_data is None:
        filename = generate_filename(args)
        if filename in os.listdir(os.path.join(input_dir)):
            data = np.load(
                os.path.join(input_dir, filename),
                allow_pickle=True)
            features = data['features']
            feature_names = data['feature_names']
            labels = data['labels']
            subjects = data['subjects']
            feature_keys = data['feature_keys']
            label_keys = data['label_keys']
        else:
            if args.task_id != 1:
                exit()
            features, feature_names, labels, subjects, feature_keys, label_keys = generate_embeddings(args)
    else:
        features = input_data['features']
        feature_names = input_data['feature_names']
        labels = input_data['labels']
        subjects = input_data['subjects']
        feature_keys = input_data['feature_keys']
        label_keys = input_data['label_keys']
    print('features.shape: ', features.shape, flush=True)
    
    # Process subjects
    for _, subj in enumerate(np.unique(subjects)):
        mask = subjects == subj
        subjects[mask] = script_subjects.subject_dict[subj]
        del mask
    
    # Train-Val-Test split
    train_mask, val_mask, test_mask = train_val_test_split(
        subjects, args)
    print('Test subject: ', np.unique(subjects[test_mask]), flush=True)
    
    # Define dataset
    train_set = EmDataset(
        X = features[train_mask],
        args = args,
    )
    
    val_set = EmDataset(
        X = features[val_mask],
        args = args,
    )
    
    test_set = EmDataset(
        X = features[test_mask],
        args = args,
    )
    
    # Define dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=args.data_config['batch_size'],
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.data_config['batch_size'],
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.data_config['batch_size'],
        num_workers=2,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


def train_val_test_split(subjects, args):
    # Create test mask
    test_mask = (subjects >= (args.task_id - 1) * 10) & (subjects < args.task_id * 10)
    
    # Create val mask
    val_subjects = np.random.choice(
        np.unique(subjects[~test_mask]), 
        size=(10, ), 
        replace=False
    )

    val_mask = np.zeros_like(test_mask, dtype=bool)
    for subj in val_subjects:
        val_mask |= (subjects == subj)
    
    # Create train mask
    train_mask = ~(val_mask | test_mask)
    return train_mask, val_mask, test_mask


class EmDataset(Dataset):
    def __init__(self, X, args):
        self.X = X
        self.args = args
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.X[idx]

    
def calculate_difference(entry1, entry2):
    """
    Calculate the difference between two (subject, day, time) entries.
    
    :param entry1: Tuple (subject, day, time)
    :param entry2: Tuple (subject, day, time)
    :return: Integer difference based on sequential ordering
    """
    subject1, day1, time1 = entry1
    subject2, day2, time2 = entry2
    
    if subject1 != subject2:
        return -np.inf
    
    # Convert (day, time) into a sequential index
    key1 = (day1 - 1) * 5 + time1
    key2 = (day2 - 1) * 5 + time2
    
    return key2 - key1

