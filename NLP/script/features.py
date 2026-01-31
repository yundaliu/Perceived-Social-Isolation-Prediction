import os
import itertools
import numpy as np
from enum import IntEnum
from collections import defaultdict


home_dir = os.path.join(
    '/home',
    'yundaliu_umass_edu',
    'social_isolation',
)

input_dir = os.path.join(
    home_dir,
    'input',
)

# # embeddings 
# emDict = np.load(
#     os.path.join(input_dir, 'EMA_embeddings.npz'),
#     allow_pickle=True)['arr_0'].item()


def load_txt_file(filepath):
    content = []
    with open(filepath, 'r') as txt:
        for line in txt.readlines():
            content.append(line.strip()) # replace('\n', ''))
    return content


# questions_dict
EMA_survey_dir = os.path.join(input_dir, 'EMA_survey')
questions_dict = defaultdict(dict)
for filename in os.listdir(EMA_survey_dir):
    if not filename.endswith('.txt'):
        continue
        
    # read file
    filepath = os.path.join(
        EMA_survey_dir,
        filename
    )
    content = load_txt_file(filepath)
    
    key = filename.split('.')[0]
    for i, line in enumerate(content):
        questions_dict[key][i + 1] = line
        

def find_preceding_keys(target_key, args): 
    # Parse target_key: subject, day, time
    subject, day, time = target_key
    
    # Generate candidates
    if 'within_day' in args.feature_config and args.feature_config['within_day']:
        all_keys = list(itertools.product(
            [subject], [day], range(1, 6)
        ))
    else:
        all_keys = list(itertools.product(
            [target_key[0]], range(1, 15), range(1, 6)
        ))
    
    if target_key not in all_keys:
        raise ValueError('Target tuple not found in sequence.')
        
    # Find index of the target tuple
    index = all_keys.index(target_key)
    
    # Get the prev preceding tuples
    if args.prev == 0:
        preceding_keys = [target_key]
    else:
        preceding_keys = all_keys[max(0, index - args.prev):index]
    return preceding_keys[::-1]


def extract_features(survey_data, survey_columns, demographics, demographics_columns, args):
    features = []
    feature_names = None
    labels = []
    subjects = []
    
    # for each feature, we want to know
    # which survey it was extracted from
    feature_keys = []
    
    # for each label, we want to know
    # which survey it was extracted from
    label_keys = []
    
    for curr_key in survey_data:
        feats, names, feat_keys = [], [], []
        
        # find preceding keys
        prev_keys = find_preceding_keys(curr_key, args)
        
        # if there are not enough keys
        # before the current key,
        # continue
        if len(prev_keys) < args.prev:
            continue
        
        for i, prev_key in enumerate(prev_keys):
            if prev_key not in survey_data:
                feats, names, feat_keys = [], [], []
                break
                
            # extract features
            if args.feature_config['use_embeddings']:
                text = generate_text(
                    survey_data[prev_key], 
                    survey_columns,
                    demographics[prev_key[0]],
                    demographics_columns,
                    i,
                    args
                )
                # # extract timestamps
                # if args.feature_config['use_timestamp']:
                #     text = str(
                #         survey_data[prev_key].ema_survey_end - 
                #         survey_data[curr_key].ema_survey_end
                #     ) + ' ' + text
                f = [text]
                n = []
            else:
                f, n = extract_features_subset(
                    survey_data[prev_key],
                    survey_columns,
                    demographics[prev_key[0]],
                    demographics_columns,
                    i,
                    args
                )
            
            # save
            feats.extend(f)
            names.extend(n)
            feat_keys.extend([prev_key] * len(f))

        if feats:
            features.append(feats)
            if feature_names is None:
                feature_names = names
            else:
                if np.any(feature_names != names):
                    raise ValueError('feature names do not match.')
            labels.append(survey_data[curr_key].isolated)
            subjects.append(curr_key[0])
            feature_keys.append(feat_keys)
            label_keys.append(curr_key)
    
    features = np.array(features)
    feature_names = np.array(feature_names)
    labels = np.array(labels)
    subjects = np.array(subjects)
    return features, feature_names, labels, subjects, feature_keys, label_keys


def extract_features_subset(data_obj, survey_columns, demographics, demographics_columns, prefix, args):
    feats = []
    names = []
    
    # Collect environmental_barrier,
    # who_is_with_you,
    # why_not_engaged_social,
    # why_not_doing_anything
    environmental_barriers = dict()
    who_is_with_you = dict()
    why_not_engaged_social = dict()
    why_not_doing_anything = dict()
    
    # Iterate survey
    for col in survey_columns:
        if col.startswith('ema') or col == 'survey_complete':
            continue
            
        # Extract value
        value = getattr(data_obj, col)
        if not isinstance(value, (int, float)):
            raise ValueError('Invalid value error.')
        
        if isinstance(value, IntEnum):
            value = value.value
            
        # Extract features
        if col == 'where_are_you':
            f, n = __feature_where_are_you(value, prefix, args)
            feats.extend(f)
            names.extend(n)
        elif col == 'how_get_where_are':
            f, n = __feature_how_get_where_are(value, prefix, args)
            feats.extend(f)
            names.extend(n)
        elif col.startswith('environmental_barriers'):
            environmental_barriers[col] = value
        elif col.startswith('who_is_with_you'):
            who_is_with_you[col] = value
        elif col.startswith('why_not_engaged_social'):
            why_not_engaged_social[col] = value
        elif col == 'what_doing_home':
            if value != 0:
                f, n = __feature_what_doing_home(value, prefix, args)
                feats.extend(f)
                names.extend(n)
        elif col == 'doing_not_home':
            if value != 0:
                f, n = __feature_doing_not_home(value, prefix, args)
                feats.extend(f)
                names.extend(n)   
        elif col.startswith('why_not_doing_anything'):
            why_not_doing_anything[col] = value
        elif col == 'isolated':
            continue
        else:
            feats.append(value)
            names.append('{:s}_{:d}'.format(col, prefix))
    
    # Extract features from
    # environmental_barrier
    f, n = __feature_environmental_barrier(environmental_barriers, prefix, args)
    feats.extend(f); names.extend(n) 
    del f, n

    # Extract features from
    # who_is_with_you
    f, n = __feature_who_is_with_you(who_is_with_you, prefix, args)
    feats.extend(f); names.extend(n) 
    del f, n

    # Extract features from
    # why_not_engaged_social
    f, n = __feature_why_not_engaged_social(why_not_engaged_social, prefix, args)
    feats.extend(f); names.extend(n) 
    del f, n

    # Extract features from
    # why_not_doing_anything
    f, n = __feature_why_not_doing_anything(why_not_doing_anything, prefix, args)
    feats.extend(f); names.extend(n) 
    del f, n
    
    if 'use_demographics' in args.feature_config and args.feature_config['use_demographics']:
        pass
    
    feats = np.array(feats)
    names = np.array(names)
    sorted_idx = np.argsort(names)
    return feats[sorted_idx], names[sorted_idx]
    

def __feature_where_are_you(value, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        # extract embeddings from where_are_you
        feats = emDict['where_are_you'][value].tolist()
        names = ['where_are_you_dim_{:03d}_{:d}'.format(i, prefix) for i in range(len(feats))]
    else:
        # dummy variable for where_are_you
        at_home = 0
        at_home_family = 0
        at_home_friend = 0
        at_work = 0
        at_medical = 0
        at_hospital = 0
        at_restaurant = 0
        at_community_center = 0
        at_store = 0
        in_vehicle = 0
        at_class = 0
        inside = 0
        outside = 0
    
        if value == 1:
            at_home = 1
        elif value == 2:
            at_home_family = 1
        elif value == 3:
            at_home_friend = 1
        elif value == 4:
            at_work = 1
        elif value == 5:
            at_medical = 1
        elif value == 6:
            at_hospital = 1
        elif value == 7:
            at_restaurant = 1
        elif value == 8:
            at_community_center = 1
        elif value == 9:
            at_store = 1
        elif value == 10:
            in_vehicle = 1
        elif value == 11:
            outside = 1
        elif value == 12:
            at_class = 1
        elif value == 13:
            inside = 1
        elif value == 14:
            outside = 1
        else:
            raise ValueError('{:d} is an invalid value for where_are_you.'.format(value))

        # add where_you_are to the feature set
        feats = []; names = []
        feats.append(at_home); names.append('at_home_{:d}'.format(prefix))
        feats.append(at_home_family); names.append('at_home_family_{:d}'.format(prefix))
        feats.append(at_home_friend); names.append('at_home_friend_{:d}'.format(prefix))
        feats.append(at_work); names.append('at_work_{:d}'.format(prefix))
        feats.append(at_medical); names.append('at_medical_{:d}'.format(prefix))
        feats.append(at_hospital); names.append('at_hospital_{:d}'.format(prefix))
        feats.append(at_restaurant); names.append('at_restaurant_{:d}'.format(prefix))
        feats.append(at_community_center); names.append('at_community_center_{:d}'.format(prefix))
        feats.append(at_store); names.append('at_store_{:d}'.format(prefix))
        feats.append(in_vehicle); names.append('in_vehicle_{:d}'.format(prefix))
        feats.append(at_class); names.append('at_class_{:d}'.format(prefix))
        feats.append(inside); names.append('inside_{:d}'.format(prefix))
        feats.append(outside); names.append('outside_{:d}'.format(prefix))
    return feats, names


def __feature_how_get_where_are(value, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        # extract embeddings from how_get_where_are
        if value in emDict['how_get_where_are']:
            feats = emDict['how_get_where_are'][value].tolist()
            names = ['how_get_where_are_dim_{:03d}_{:d}'.format(i, prefix) for i in range(len(feats))]
        else:
            emLen = len(emDict['how_get_where_are'][1])
            feats = [0 for _ in range(emLen)]
            names = ['how_get_where_are_dim_{:03d}_{:d}'.format(i, prefix) for i in range(emLen)]
    else:
        # dummy variable for how_get_where_are
        walked = 0
        ride = 0
        bus = 0
        drive = 0
        # other = 0

        if value == 0:
            pass
        elif value == 1:
            walked = 1
        elif value == 2:
            ride = 1
        elif value == 3:
            bus = 1
        elif value == 4:
            drive = 1
        elif value == 5:
            # other = 1
            pass
        else:
            raise ValueError('{:d} is an invalid value for how_get_where_are.'.format(value))

        # add where_you_are to the feature set
        feats = []; names = []
        feats.append(walked); names.append('walked_{:d}'.format(prefix))
        feats.append(ride); names.append('ride_{:d}'.format(prefix))
        feats.append(bus); names.append('bus_{:d}'.format(prefix))
        feats.append(drive); names.append('drive_{:d}'.format(prefix))
        # feats.append(other); names.append('other_{:d}'.format(prefix))
    return feats, names


def __feature_what_doing_home(value, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        # extract embeddings from what_doing_home/doing_not_home
        feats = emDict['what_doing_home'][value].tolist()
        names = ['activity_dim_{:03d}_{:d}'.format(i, prefix) for i in range(len(feats))]
    else:
        # dummy variable for activity
        ADL = 0
        cognitively_stimulating = 0
        IADL = 0
        passive_leisure = 0
        physical_activity = 0
        social_activity = 0
        vocational_activity = 0

        if 1 <= value <= 3:
            ADL = 1
        elif 4 <= value <= 11:
            cognitively_stimulating = 1
        elif 12 <= value <= 20:
            IADL = 1
        elif 21 <= value <= 25:
            passive_leisure = 1
        elif 26 <= value <= 29:
            physical_activity = 1
        elif 30 <= value <= 42:
            social_activity = 1
        elif 43 <= value <= 46:
            vocational_activity = 1
        else:
            raise ValueError('Invalid value for activity.')


        # add activity to the feature set
        feats = []; names = []
        feats.append(ADL); names.append('ADL_{:d}'.format(prefix))
        feats.append(cognitively_stimulating); names.append('cognitively_stimulating_{:d}'.format(prefix))
        feats.append(IADL); names.append('IADL_{:d}'.format(prefix))
        feats.append(passive_leisure); names.append('passive_leisure_{:d}'.format(prefix))
        feats.append(physical_activity); names.append('physical_activity_{:d}'.format(prefix))
        feats.append(social_activity); names.append('social_activity_{:d}'.format(prefix))
        feats.append(vocational_activity); names.append('vocational_activity_{:d}'.format(prefix))
    return feats, names
    
    
def __feature_doing_not_home(value, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        # extract embeddings from what_doing_home/doing_not_home
        feats = emDict['doing_not_home'][value].tolist()
        names = ['activity_dim_{:03d}_{:d}'.format(i, prefix) for i in range(len(feats))]
    else:
        # dummy variable for activity
        ADL = 0
        cognitively_stimulating = 0
        IADL = 0
        passive_leisure = 0
        physical_activity = 0
        social_activity = 0
        vocational_activity = 0

        if 1 <= value <= 3:
            ADL = 1
        elif 4 <= value <= 11:
            cognitively_stimulating = 1
        elif 12 <= value <= 20:
            IADL = 1
        elif 21 <= value <= 25:
            passive_leisure = 1
        elif 26 <= value <= 29:
            physical_activity = 1
        elif 30 <= value <= 42:
            social_activity = 1
        elif 43 <= value <= 46:
            vocational_activity = 1
        else:
            raise ValueError('Invalid value for activity.')


        # add activity to the feature set
        feats = []; names = []
        feats.append(ADL); names.append('ADL_{:d}'.format(prefix))
        feats.append(cognitively_stimulating); names.append('cognitively_stimulating_{:d}'.format(prefix))
        feats.append(IADL); names.append('IADL_{:d}'.format(prefix))
        feats.append(passive_leisure); names.append('passive_leisure_{:d}'.format(prefix))
        feats.append(physical_activity); names.append('physical_activity_{:d}'.format(prefix))
        feats.append(social_activity); names.append('social_activity_{:d}'.format(prefix))
        feats.append(vocational_activity); names.append('vocational_activity_{:d}'.format(prefix))
    return feats, names
    
    
def __feature_environmental_barrier(environmental_barriers, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        emLen = len(emDict['environmental_barriers'][1])
        feats = []
        names = ['environmental_barrier_dim_{:03d}_{:d}'.format(i, prefix) for i in range(emLen)]
        for key, val in environmental_barriers.items():
            if val == 1:
                feats.append(emDict['environmental_barriers'][int(key.split('___')[1])])
        if len(feats):
            feats = np.array(feats).mean(axis=0).tolist()
        else:
            feats = [0 for _ in range(emLen)]
    else:
        feats = []; names = []
        for key, val in environmental_barriers.items():
            if key == 'environmental_barriers___1':
                continue
            
            feats.append(val)
            names.append('{:s}_{:d}'.format(key, prefix))
    return feats, names


def __feature_who_is_with_you(who_is_with_you, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        emLen = len(emDict['who_is_with_you'][1])
        feats = []
        names = ['who_is_with_you_dim_{:03d}_{:d}'.format(i, prefix) for i in range(emLen)]
        for key, val in who_is_with_you.items():
            if val == 1:
                feats.append(emDict['who_is_with_you'][int(key.split('___')[1])])
        if len(feats):
            feats = np.array(feats).mean(axis=0).tolist()
        else:
            feats = [0 for _ in range(emLen)]
    else:
        feats = []; names = []
        for key, val in who_is_with_you.items():
            feats.append(val)
            names.append('{:s}_{:d}'.format(key, prefix))
    return feats, names


def __feature_why_not_engaged_social(why_not_engaged_social, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        emLen = len(emDict['why_not_engaged_social'][1])
        feats = []
        names = ['why_not_engaged_social_dim_{:03d}_{:d}'.format(i, prefix) for i in range(emLen)]
        for key, val in why_not_engaged_social.items():
            if val == 1:
                feats.append(emDict['why_not_engaged_social'][int(key.split('___')[1])])
        if len(feats):
            feats = np.array(feats).mean(axis=0).tolist()
        else:
            feats = [0 for _ in range(emLen)]
    else:
        feats = []; names = []
        for key, val in why_not_engaged_social.items():
            feats.append(val)
            names.append('{:s}_{:d}'.format(key, prefix))
    return feats, names


def __feature_why_not_doing_anything(why_not_doing_anything, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        emLen = len(emDict['why_not_doing_anything'][1])
        feats = []
        names = ['why_not_doing_anything_dim_{:03d}_{:d}'.format(i, prefix) for i in range(emLen)]
        for key, val in why_not_doing_anything.items():
            if val == 1:
                feats.append(emDict['why_not_doing_anything'][int(key.split('___')[1])])
        if len(feats):
            feats = np.array(feats).mean(axis=0).tolist()
        else:
            feats = [0 for _ in range(emLen)]
    else:
        feats = []; names = []
        for key, val in why_not_doing_anything.items():
            feats.append(val)
            names.append('{:s}_{:d}'.format(key, prefix))
    return feats, names


def __feature_age(age, prefix, args):
    if 'use_embeddings' in args.feature_config and args.feature_config['use_embeddings']:
        feats


def aggregate_features(features, feature_names, args):
    n_groups = args.prev
    if n_groups == 1:
        return features, feature_names
    
    _, n_features = features.shape
    n_features_per_group = n_features // n_groups
    
    # save new features and feature_names
    agg_features = []
    agg_feature_names = []
    
    for i in range(n_features_per_group):
        # extract features
        indices = i + np.arange(n_groups) * n_features_per_group
        feat = features[:, indices]
        name = '_'.join(feature_names[i].split('_')[:-1])
        
        # mean
        agg_features.append(np.mean(feat, axis=1, keepdims=True))
        agg_feature_names.append('{:s}_mean'.format(name))
        
        # std
        if 'dim' not in name:
            agg_features.append(np.std(feat, axis=1, keepdims=True))
            agg_feature_names.append('{:s}_std'.format(name))
        
        if name in ['anxiety', 'appetite', 'cheerful',  
                    'concentrating', 'confidence_social', 
                    'depressed', 'engaged_activity', 
                    'help',  'learning_new',
                    'little_interest', 'mindfullness',  
                    'pain', 'perform_activity', 
                    'satisfaction_social', 'satisfied_activity',
                    'slow_restless', 'stress',
                    'success_social', 'times_socialized', 
                    'tired', 'worthless']:
            # max
            agg_features.append(np.max(feat, axis=1, keepdims=True))
            agg_feature_names.append('{:s}_max'.format(name))
        
            # min
            agg_features.append(np.min(feat, axis=1, keepdims=True))
            agg_feature_names.append('{:s}_min'.format(name))
        
        if args.feature_config['add_features']:
            ##################################
            # Percentage of prompts for which
            # a nonzero score was reported
            ##################################
            if name in ['ADL', 'IADL', 'cognitively_stimulating',
                        'passive_leisure', 'physical_activity',
                        'social_activity', 'vocational_activity',
                        'at_class', 'at_community_center', 'at_home', 
                        'at_home_family', 'at_home_friend', 'at_hospital', 
                        'at_medical', 'at_restaurant', 'at_store', 
                        'at_work', 'in_vehicle', 'inside', 'outside', 
                        'bus', 'drive', 'ride', 'walked', 
                        'environmental_barriers___10', 
                        'environmental_barriers___2', 
                        'environmental_barriers___3', 
                        'environmental_barriers___4', 
                        'environmental_barriers___5',
                        'environmental_barriers___6', 
                        'environmental_barriers___7', 
                        'environmental_barriers___8', 
                        'environmental_barriers___9',
                        'who_is_with_you___10', 'who_is_with_you___1', 
                        'who_is_with_you___2', 'who_is_with_you___3', 
                        'who_is_with_you___4', 'who_is_with_you___5', 
                        'who_is_with_you___6', 'who_is_with_you___7', 
                        'who_is_with_you___8', 'who_is_with_you___9', 
                        'why_not_doing_anything___10',
                        'why_not_doing_anything___1', 'why_not_doing_anything___2',
                        'why_not_doing_anything___3', 'why_not_doing_anything___4',
                        'why_not_doing_anything___5', 'why_not_doing_anything___6',
                        'why_not_doing_anything___7', 'why_not_doing_anything___8',
                        'why_not_doing_anything___9', 'why_not_engaged_social___1',
                        'why_not_engaged_social___2', 'why_not_engaged_social___3',
                        'why_not_engaged_social___4', 'why_not_engaged_social___5',
                        'why_not_engaged_social___6', 'why_not_engaged_social___7']:
                agg_features.append(np.mean(feat != 0, axis=1, keepdims=True))
                agg_feature_names.append('{:s}_percentage'.format(name))
            elif name in ['anxiety', 'appetite', 'cheerful',
                          'concentrating', 'depressed',
                          'isolated', 'learning_new',
                          'little_interest', 'mindfullness',
                          'pain', 'slow_restless','stress',
                          'tired', 'worthless']:
                agg_features.append(np.mean(feat != 1, axis=1, keepdims=True))
                agg_feature_names.append('{:s}_percentage'.format(name))
                
           
            ##################################
            # RMS of successive differences
            ##################################
            if name in ['anxiety', 'appetite', 'cheerful',  
                        'concentrating', 'confidence_social', 
                        'depressed', 'engaged_activity', 
                        'help',  'learning_new',
                        'little_interest', 'mindfullness',  
                        'pain', 'perform_activity', 
                        'satisfaction_social', 'satisfied_activity',
                        'slow_restless', 'stress',
                        'success_social', 'times_socialized', 
                        'tired', 'worthless']:
                agg_features.append(np.sqrt(np.mean(np.square(feat[:, 1:] - feat[:, :-1]), axis=1, keepdims=True)))
                agg_feature_names.append('{:s}_RMSSD'.format(name))

            ##################################
            # Coefficient Covariance
            ##################################
            if name in ['anxiety', 'appetite', 'cheerful',  
                        'concentrating', 'confidence_social', 
                        'depressed', 'engaged_activity', 
                        'help',  'learning_new',
                        'little_interest', 'mindfullness',  
                        'pain', 'perform_activity', 
                        'satisfaction_social', 'satisfied_activity',
                        'slow_restless', 'stress',
                        'success_social', 'times_socialized', 
                        'tired', 'worthless']:
                agg_features.append(np.mean(feat, axis=1, keepdims=True) / np.std(feat, axis=1, keepdims=True))
                agg_feature_names.append('{:s}_CV'.format(name))


            ##################################
            # Mean Successive Variability
            ##################################  
            if name in ['anxiety', 'appetite', 'cheerful',  
                        'concentrating', 'confidence_social', 
                        'depressed', 'engaged_activity', 
                        'help',  'learning_new',
                        'little_interest', 'mindfullness',  
                        'pain', 'perform_activity', 
                        'satisfaction_social', 'satisfied_activity',
                        'slow_restless', 'stress',
                        'success_social', 'times_socialized', 
                        'tired', 'worthless']:
                agg_features.append(np.mean(np.abs(feat[:, 1:] - feat[:, :-1]), axis=1, keepdims=True))
                agg_feature_names.append('{:s}_ASV'.format(name))
    
    return np.hstack(agg_features), np.array(agg_feature_names)


def process_features(features, feature_names, args):
    if not args.feature_config['aggregate_features'] and not args.feature_config['add_features']:
        return features, feature_names
    
    # Create variables to hold new features
    agg_features, agg_feature_names = None, None
    
    ###########################################
    # Aggregate features
    ###########################################
    if args.feature_config['aggregate_features'] or args.feature_config['add_features']:
        agg_features, agg_feature_names = aggregate_features(features, feature_names, args)
        
    ###########################################
    # Combine features
    ###########################################
    if 'combine_features' in args.feature_config and args.feature_config['combine_features']:
        agg_features = np.hstack((features, agg_features))
        agg_feature_names = np.hstack((feature_names, agg_feature_names))
        
    return agg_features, agg_feature_names


def remove_invalid_features(features, labels, feature_names, subjects, axis=0):
    if axis == 0:
        invalid_mask = np.any(np.isnan(features), axis=axis) | np.any(np.isinf(features), axis=axis)
        features = features[:, ~invalid_mask]
        feature_names = feature_names[~invalid_mask]
        # print(f'Removed {invalid_mask.sum()} features.')
        # print(f'Extracted {(~invalid_mask).sum()} features.')
    else:
        invalid_mask = np.any(np.isnan(features), axis=axis) | np.any(np.isinf(features), axis=axis)
        features = features[~invalid_mask]
        labels = labels[~invalid_mask]
        subjects = subjects[~invalid_mask]
        # feature_names = feature_names[~invalid_mask]
        # print(f'Removed {invalid_mask.sum()} features.')
        # print(f'Extracted {(~invalid_mask).sum()} features.')
    return features, labels, feature_names, subjects


def generate_text(data_obj, survey_columns, demographics, demographics_columns, prefix, args):
    ans = []
    
    # where_are_you and how_get_where_are
    ans.append(__extract_where_are_you(data_obj))
    
    # environmental_barriers
    ans.append(__extract_text_environmental_barriers(data_obj))
    
    # the num. of interactions
    ans.append(__extract_times_socialized(data_obj))
    
    # who is with me
    ans.append(__extract_who_is_with_you(data_obj))
    
    # activity
    if data_obj.what_doing_home != 0:
        ans.append(__extract_activity(data_obj, 'what_doing_home'))
    elif data_obj.doing_not_home != 0:
        ans.append(__extract_activity(data_obj, 'doing_not_home'))
            
    # # secondary conditions
    # ans.append(__extract_secondary_conditions(data_obj))
    
    # pain
    ans.append(__extract_pain(data_obj))
    
    # stress
    ans.append(__extract_stress(data_obj))
    
    # tired
    ans.append(__extract_tired(data_obj))
    
    # depressed
    ans.append(__extract_depressed(data_obj))
    
    # worthless
    ans.append(__extract_worthless(data_obj))
    
    # concentrating
    ans.append(__extract_concentrating(data_obj))
    
    # learning_new
    ans.append(__extract_learning_new(data_obj))
    
    # anxiety
    ans.append(__extract_anxiety(data_obj))
    
    # mindfullness
    ans.append(__extract_mindfullness(data_obj))
    
    # little_interest
    ans.append(__extract_little_interest(data_obj))
    
    # appetite
    ans.append(__extract_appetite(data_obj))
    
    # slow_restless
    ans.append(__extract_slow_restless(data_obj))
    
    # cheerful
    ans.append(__extract_cheerful(data_obj))
    
    # demographics
    if 'use_demographics' in args.feature_config and args.feature_config['use_demographics']:
        ans.extend(__extract_demographics(demographics))
        
    # social isolation
    if 'use_isolation' in args.feature_config and args.feature_config['use_isolation']:
        ans.append(__extract_isolated(data_obj))
    
    # timestamp
    if 'use_timestamp' in args.feature_config and args.feature_config['use_timestamp']:
        # extract timestamps
        ans.append('Survey start time was {:s}'.format(str(data_obj.ema_survey_start)))
        ans.append('Survey end time was {:s}'.format(str(data_obj.ema_survey_end)))
    
    # return ' '.join(ans)
    return ans

    
def __extract_where_are_you(data_obj):
    # where_are_you
    question1 = 'where_are_you'
    value1 = getattr(data_obj, question1)
    txt = 'I am at {:s} right now.'.format(questions_dict[question1][value1])
    
    # If the person was not at home,
    # extract how_get_where_are
    if value1 != 1:
        question2 = 'how_get_where_are'
        value2 = getattr(data_obj, question2)
        txt += ' I {:s}.'.format(questions_dict[question2][value2])
        del question2, value2
    return txt
        
    
def __extract_text_environmental_barriers(data_obj):
    # environmental_barriers
    question1 = 'environmental_barriers___1'
    value1 = getattr(data_obj, question1)
    if value1 == 1:
        return 'None of these things have gotten in the way.'
    
    txt = 'The following things have gotten in the way: '
    for n in np.arange(2, 11, dtype=int):
        question2 = 'environmental_barriers___{:d}'.format(n)
        value2 = getattr(data_obj, question2)
        if value2 == 1:
            txt += ' {:s},'.format(questions_dict[question2.split('___')[0]][n])
        del question2, value2
    txt = txt[:-1] + '.'
    return txt


def __extract_times_socialized(data_obj):
    # times_socialized
    question = 'times_socialized'
    value = getattr(data_obj, question)
    txt = 'I have {:s} with someone else since the last alarm.'.format(
        questions_dict[question][value])
    return txt


def __extract_who_is_with_you(data_obj):
    # who is with you 
    txt = 'I am with'
    for n in np.arange(1, 11, dtype=int):
        question = 'who_is_with_you___{:d}'.format(n)
        value = getattr(data_obj, question)
        if value == 1:
            txt += ' {:s},'.format(questions_dict[question.split('___')[0]][n])
    txt = txt[:-1] + '.'
    del question, value
    
    # confidence_social
    question = 'confidence_social'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The confidence level in my interaction with the person(s) I am with is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # satisfaction_social
    question = 'satisfaction_social'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The satisfaction level in my interaction with the person(s) I am with is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # success_social
    question = 'success_social'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The success level in my interaction with the person(s) I am with is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # why_not_engaged_social
    why_not_engaged_social = []
    for n in np.arange(1, 7, dtype=int):
        question = 'why_not_engaged_social___{:d}'.format(n)
        value = getattr(data_obj, question)
        if value == 1:
            why_not_engaged_social.append(questions_dict[question.split('___')[0]][n])
        del question, value
    if why_not_engaged_social:
        txt += ' I am not doing anything social because ' + ', '.join(why_not_engaged_social) + '.'
    
    return txt


def __extract_activity(data_obj, question):
    # what_doing_home/doing_not_home
    value = getattr(data_obj, question)
    txt = 'I am doing {:s}.'.format(
        questions_dict[question][value])

    # help
    question = 'help'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The help I am getting from someone else while doing this activity is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # perform_activity
    question = 'perform_activity'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The activity performance is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # satisfied_activity
    question = 'satisfied_activity'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The satisfaction in the activity is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # engaged_activity
    question = 'engaged_activity'
    value = getattr(data_obj, question)
    if value != 0:
        txt += ' The engagement in the activity is {:d}.'.format(
            int(value)
        )
    del question, value
    
    # why_not_doing_anything
    why_not_doing_anything = []
    for n in np.arange(1, 11, dtype=int):
        question = 'why_not_doing_anything___{:d}'.format(n)
        value = getattr(data_obj, question)
        if value == 1:
            why_not_doing_anything.append(questions_dict[question.split('___')[0]][n])
        del question, value
    if why_not_doing_anything:
        txt += ' I am not doing anything because ' + ', '.join(why_not_doing_anything) + '.'
        
    return txt


def __extract_pain(data_obj):
    # pain
    question = 'pain'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I have no pain.'
    else:
        txt = 'My pain level is {:s}.'.format(questions_dict[question][value])
    return txt


def __extract_stress(data_obj):
    # stress
    question = 'stress'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel nervous or stressed at all.'
    else:
        txt = 'I feel {:s} nervous and stressed.'.format(questions_dict['likert'][value])
    del question, value
    return txt
    
    
def __extract_tired(data_obj):
    # tired
    question = 'tired'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel tired at all.'
    else:
        txt = 'I feel {:s} tired.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_depressed(data_obj):
    # depressed
    question = 'depressed'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel depressed at all.'
    else:
        txt = 'I feel {:s} depressed.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_worthless(data_obj):
    # worthless
    question = 'worthless'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel worthless at all.'
    else:
        txt = 'I feel {:s} worthless.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_concentrating(data_obj):
    # concentrating
    question = 'concentrating'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not have trouble concentrating at all.'
    else:
        txt = 'I have {:s} trouble concentrating.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_learning_new(data_obj):
    # learning_new
    question = 'learning_new'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not have difficulty learning new tasks or instructions at all.'
    else:
        txt = 'I have {:s} difficulty learning new tasks or instructions.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_anxiety(data_obj):
    # anxiety
    question = 'anxiety'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel uneasy at all.'
    else:
        txt = 'I feel {:s} uneasy.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_mindfullness(data_obj):
    # mindfullness
    question = 'mindfullness'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'It is not easy for me to keep track of my thoughts and feelings.'
    else:
        txt = 'It is {:s} easy for me to keep track of my thoughts and feelings.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_little_interest(data_obj):
    # little_interest
    question = 'little_interest'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not have little interest in doing things at all.'
    else:
        txt = 'I have {:s} little interest in doing things at all.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_appetite(data_obj):
    # appetite
    question = 'appetite'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not have a poor appetite at all.'
    else:
        txt = 'I have a {:s} poor appetite.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_slow_restless(data_obj):
    # slow_restless
    question = 'slow_restless'
    value = getattr(data_obj, question)
    txt = 'My moving/speaking is so slow or restless that other people have noticed {:s}.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_cheerful(data_obj):
    # cheerful
    question = 'cheerful'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel cheerful at all.'
    else:
        txt = 'I feel {:s} cheeful.'.format(questions_dict['likert'][value])
    del question, value
    return txt


def __extract_isolated(data_obj):
    # cheerful
    question = 'isolated'
    value = getattr(data_obj, question)
    if value == 1:
        txt = 'I do not feel isolated at all.'
    else:
        txt = 'I feel {:s} isolated.'.format(questions_dict['likert'][value])
    del question, value
    return txt
    

def __extract_demographics(data_obj):
    txt = []
    
    # age
    question = 'age'
    value = getattr(data_obj, question)
    txt.append('I am {:d} years old.'.format(value))
    del question, value
    
    # gender
    question = 'gender'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I am male.')
    else:
        txt.append('I am female.')
    del question, value
    
    # race
    question = 'race'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I am black.')
    elif value == 2:
        txt.append('I am white.')
    elif value == 3:
        txt.append('I am asian.')
    elif value == 4:
        txt.append('I am american indian or alaska native.')
    elif value == 5:
        txt.append('I am native hawaiian or other pacific islanders.')
    del question, value
    
    # marital status
    question = 'marital'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I am married.')
    elif value == 2:
        txt.append('I am divorced.')
    elif value == 3:
        txt.append('I am widowed.')
    elif value == 4:
        txt.append('I am separated.')
    elif value == 5:
        txt.append('I have never been married.')
    elif value == 6:
        txt.append('I am a member of an unmarried couple.')
    del question, value
        
    # residential status
    question = 'resident'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I live alone.')
    else:
        t = []
        question = 'resident_notalone___1'
        value = getattr(data_obj, question)
        if value == 1:
            t.append('spouse/partner')
        del question, value
        
        question = 'resident_notalone___2'
        value = getattr(data_obj, question)
        if value == 1:
            t.append('child/grandchild')
        del question, value
        
        question = 'resident_notalone___3'
        value = getattr(data_obj, question)
        if value == 1:
            t.append('other relatives')
        del question, value
        
        question = 'resident_notalone___4'
        value = getattr(data_obj, question)
        if value == 1:
            t.append('friends')
        del question, value
        
        question = 'resident_notalone___5'
        value = getattr(data_obj, question)
        if value == 1:
            t.append('pets')
        del question, value
        
        question = 'resident_notalone___6'
        value = getattr(data_obj, question)
        if value == 1:
            resident_other = getattr(data_obj, 'resident_others')
            if isinstance(resident_other, str):
                t.append(resident_other.lower())
        txt.append('I live with ' + ','.join(t) + '.')
        del question, value
        
    # employement
    question = 'employment'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I am employed full-time at my usual job.')
    elif value == 2:
        txt.append('I am employed full-time at my usual employer in a modified job.')
    elif value == 3:
        txt.append('I am employed full-time at my usual employer in a different job.')
    elif value == 4:
        txt.append('I am employed full-time at a new employer in a modified job.')
    elif value == 5:
        txt.append('I am employed full-time at a new employer in a different job.')
    elif value == 6:
        txt.append('I am employed part-time at my usual job.')
    elif value == 7:
        txt.append('I am employed part-time at my usual employer in a modified job.')
    elif value == 8:
        txt.append('I am employed part-time at my usual employer in a different job.')
    elif value == 9:
        txt.append('I am employed part-time at a new employer in a modified job.')
    elif value == 10:
        txt.append('I am employed part-time at a new employer in a different job.')
    elif value == 11:
        txt.append('I am employed in a volunteer position.')
    elif value == 0:
        txt.append('I am not employed at all.')
    
    # workhr
    question = 'workhr'
    value = getattr(data_obj, question)
    if not np.isnan(value):
        txt.append('I work {:.1f} hours per week.'.format(value))
    else:
        txt.append('I work 0 hours per week.')
    
    # income_person
    question = 'income_person'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('My total personal income from all sources is $0 to $14,999.')
    elif value == 2:
        txt.append('My total personal income from all sources is $15,000 to $34,999.')
    elif value == 3:
        txt.append('My total personal income from all sources is $35,000 to $54,999.')
    elif value == 4:
        txt.append('My total personal income from all sources is $55,000 to $74,999.')
    elif value == 5:
        txt.append('My total personal income from all sources is $75,000 or more.')
        
    # income_household
    question = 'income_household'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('My total household income from all sources is $0 to $14,999.')
    elif value == 2:
        txt.append('My total household income from all sources is $15,000 to $34,999.')
    elif value == 3:
        txt.append('My total household income from all sources is $35,000 to $54,999.')
    elif value == 4:
        txt.append('My total household income from all sources is $55,000 to $74,999.')
    elif value == 5:
        txt.append('My total household income from all sources is $75,000 or more.')
    
    # financial responsibilities
    question = 'financial'
    value = getattr(data_obj, question)
    if value == 1:
        txt.append('I bear primary or co-equal responsibility.')
    elif value == 2:
        txt.append('I bear only partial responsibility.')
    elif value == 3:
        txt.append('I am dependent in the home of my relative, or living in a group home.')
    elif value == 4:
        txt.append('I am living in a residential treatment facility.')
        
    return txt
    
# def __extract_secondary_conditions(data_obj):
#     # pain
#     question = 'pain'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt = 'I have no pain.'
#     else:
#         txt = 'My pain level is {:s}.'.format(questions_dict[question][value])
#     del question, value
    
#     # stress
#     question = 'stress'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel nervous or stressed at all.'
#     else:
#         txt += ' I feel {:s} nervous and stressed.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # tired
#     question = 'tired'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel tired at all.'
#     else:
#         txt += ' I feel {:s} tired.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # depressed
#     question = 'depressed'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel depressed at all.'
#     else:
#         txt += ' I feel {:s} depressed.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # worthless
#     question = 'worthless'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel worthless at all.'
#     else:
#         txt += ' I feel {:s} worthless.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # concentrating
#     question = 'concentrating'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not have trouble concentrating at all.'
#     else:
#         txt += ' I have {:s} trouble concentrating.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # learning_new
#     question = 'learning_new'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not have difficulty learning new tasks or instructions at all.'
#     else:
#         txt += ' I have {:s} difficulty learning new tasks or instructions.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # anxiety
#     question = 'anxiety'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel uneasy at all.'
#     else:
#         txt += ' I feel {:s} uneasy.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # mindfullness
#     question = 'mindfullness'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' It is not easy for me to keep track of my thoughts and feelings.'
#     else:
#         txt += ' It is {:s} easy for me to keep track of my thoughts and feelings.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # little_interest
#     question = 'little_interest'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not have little interest in doing things at all.'
#     else:
#         txt += ' I have {:s} little interest in doing things at all.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # appetite
#     question = 'appetite'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not have a poor appetite at all.'
#     else:
#         txt += ' I have a {:s} poor appetite.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # slow_restless
#     question = 'slow_restless'
#     value = getattr(data_obj, question)
#     txt += ' My moving/speaking is so slow or restless that other people have noticed {:s}.'.format(questions_dict['likert'][value])
#     del question, value
    
#     # cheerful
#     question = 'cheerful'
#     value = getattr(data_obj, question)
#     if value == 1:
#         txt += ' I do not feel cheerful at all.'
#     else:
#         txt += ' I feel {:s} cheeful.'.format(questions_dict['likert'][value])
#     del question, value
    
#     return txt