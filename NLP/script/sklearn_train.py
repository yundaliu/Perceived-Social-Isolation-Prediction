import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import make_scorer, SCORERS, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from gpboost import GPBoostClassifier
# from catboost import CatBoostClassifier

import statsmodels.genmod.bayes_mixed_glm as sms

import warnings
warnings.filterwarnings('ignore')

        
class AUROCThreshold():
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        
    def fit(self, X, y):
        n_features = X.shape[1]
        feature_indices = np.arange(n_features, dtype=int)
        exclusion_mask = np.zeros((n_features, ), dtype=bool)
        for i in range(n_features):
            exclusion_mask[i] = roc_auc_score(y, X[:, i]) <= self.threshold
        self.feature_to_select = feature_indices[~exclusion_mask]
        
        print('{:d} features were kept.'.format(len(self.feature_to_select)), flush=True)
        return self
    
    def transform(self, X):
        return X[:, self.feature_to_select]
    
    
class AUPRCThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        
    def fit(self, X, y):
        n_features = X.shape[1]
        feature_indices = np.arange(n_features, dtype=int)
        
        # calculate auprc for each feature
        auprc = np.zeros((n_features, ))
        for i in range(n_features):
            # precision, recall, _ = precision_recall_curve(
            #     y, X[:, i], sample_weight=compute_sample_weight('balanced', y))
            # auprc[i] = auc(recall, precision)
            auprc[i] = average_precision_score(
                y, X[:, i], sample_weight=compute_sample_weight('balanced', y)
            )
        
        # rank features based on auprc
        self.feature_to_select = feature_indices[auprc > self.threshold]
        
#         mask = auprc > self.threshold
#         auprc = auprc[mask]
#         feature_indices = feature_indices[mask]
#         sorted_idx = np.argsort(auprc)[::-1]
#         feature_indices = feature_indices[sorted_idx]
        
#         self.feature_to_select = []
#         for feature_idx in feature_indices:
#             if not self.feature_to_select:
#                 self.feature_to_select.append(feature_idx)
#                 continue
#             corr_with_selected = spearmanr(X[:, self.feature_to_select], X[:, feature_idx])[0]
#             if np.all(abs(corr_with_selected)) < 0.7:
#                 self.feature_to_select.append(feature_idx)
        
        # print('{:d} features were kept.'.format(len(self.feature_to_select)), flush=True)
        return self
    
    def transform(self, X):
        if len(self.feature_to_select) == 0:
            return np.full((X.shape[0], 1), np.nan)
        return X[:, self.feature_to_select]


class SafeLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, class_weight=None, n_jobs=None, penalty='l2', random_state=None, solver='lbfgs'):
        self.C = C
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.penalty = penalty
        self.random_state = random_state
        self.solver = solver
    
        self.is_valid_ = True

    def fit(self, X, y):
        # If features contain np.nan
        # skip fitting
        if np.any(np.isnan(X)):
            self.is_valid_ = False
            return self
        
        # Define model
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            penalty=self.penalty,
            random_state=self.random_state,
            solver=self.solver
        )
        self.model.fit(X, y)
        self.is_valid_ = True
        return self

    def predict(self, X):
        if not self.is_valid_:
            return np.full((X.shape[0], ), np.nan)  # Default output when no features are selected
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_valid_:
            return np.full((X.shape[0], ), np.nan)  # Default output when no features are selected
        return self.model.predict_proba(X)
    

# class AUROCSequentialFeatureSelector():
#     def __init__(self, threshold=0.7, 
#                  n_features_to_select='auto', 
#                  grid_search_scoring='roc_auc',
#                  model='LogisticRegression'):
#         self.threshold = threshold
#         self.n_features_to_select = n_features_to_select
#         self.grid_search_scoring = grid_search_scoring
#         self.model = model
        
#     def fit(self, X, y):
#         # Use AUROC to select features first
#         n_features = X.shape[1]
#         feature_indices = np.arange(n_features, dtype=int)
#         exclusion_mask = np.zeros((n_features, ), dtype=bool)
#         for i in range(n_features):
#             exclusion_mask[i] = roc_auc_score(y, X[:, i]) <= self.threshold
#         self.feature_to_select = feature_indices[~exclusion_mask]
        
#         # Use SequentialFeatureSelector
#         if self.model == 'LogisticRegression':
#             self.fsel = SequentialFeatureSelector(
#                 LogisticRegression(class_weight='balanced', random_state=0, n_jobs=-1),
#                 n_features_to_select=self.n_features_to_select,
#                 scoring=self.grid_search_scoring,
#                 cv=5,
#                 n_jobs=-1
#             ).fit(X[:, self.feature_to_select], y)
#         elif self.model == 'LGBMClassifier':
#             self.fsel = SequentialFeatureSelector(
#                 LGBMClassifier(class_weight='balanced', random_state=0, n_jobs=-1, verbosity=-1),
#                 n_features_to_select=self.n_features_to_select,
#                 scoring=self.grid_search_scoring,
#                 cv=5,
#                 n_jobs=-1
#             ).fit(X[:, self.feature_to_select], y)
        
#         return self
    
#     def transform(self, X):
#         return self.fsel.transform(X[:, self.feature_to_select])

# class AUROCQuestionSelector():
#     def __init__(self, threshold=0.7, n_questions_to_select=1):
#         self.threshold = threshold
#         self.n_questions_to_select = n_questions_to_select
    
#     def fit(self, X, y, feature_names):
#         # for each question
#         # calculate auroc of associated features
#         question_dict = defaultdict(list)
#         for idx, name in enumerate(feature_names):
#             # process name
#             name = '_'.join(name.split('_')[:-1])
            
#             # calculate and save auroc
#             question_dict[name].append(roc_auc_score(y, X[:, idx]))
            
#         # find the top K questions
#         question_scores = [[key, np.mean(np.array(value) > self.threshold)] for key, value in question_dict.items()]
#         question_scores = sorted(question_scores, key=lambda x: x[1])[::-1]
#         question_scores = question_scores[:self.n_questions_to_select]
        
#         # Among the top K questions,
#         # find features with auroc greater
#         # than the threshold
#         candidate_questions = set([q for q, _ in question_scores])
#         self.feature_to_select = []
#         for idx, name in enumerate(feature_names):
#             # process name
#             name = '_'.join(name.split('_')[:-1])
#             if name not in candidate_questions:
#                 continue
                
#             # calculate and save auroc
#             if roc_auc_score(y, X[:, idx]) <= self.threshold:
#                 continue
            
#             self.feature_to_select.append(idx)
        
#         return self
    
#     def transform(self, X):
#         return X[:, self.feature_to_select]
        
        
def feature_selection(X_train, y_train, groups_train, feature_names, args):
    if args.model_config['feature_selection'] == 'AUROCThreshold':
        fsel = AUROCThreshold(args.model_config['feature_selection_thresh']).fit(X_train, y_train)
    elif args.model_config['feature_selection'] == 'AUPRCThreshold':
        fsel = AUPRCThreshold(args.model_config['feature_selection_thresh']).fit(X_train, y_train)
    return fsel
        

def train(X_train, y_train, groups_train, args):
    if args.model_config['model'] == 'RandomForestClassifier':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)
        
        # Train Model
        clf = RandomForestClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'LogisticRegression':
        # Gridsearch 
        grid_search = GridSearchCV(
            estimator=LogisticRegression(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = LogisticRegression(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'XGBoost':
        # Gridsearch 
        grid_search = GridSearchCV(
            estimator=XGBClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = XGBClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'LGBMClassifier':
        # Gridsearch 
        grid_search = GridSearchCV(
            estimator=LGBMClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = LGBMClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'RidgeClassifier':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=RidgeClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = RidgeClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'SGDClassifier':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=SGDClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = SGDClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'GPBoostClassifier':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=GPBoostClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = GPBoostClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'CatBoostClassifier':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=CatBoostClassifier(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = CatBoostClassifier(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'LinearSVC':
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=LinearSVC(), 
            param_grid=args.model_config['param_grid'], 
            cv=5, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)

        # Train Model
        clf = LinearSVC(**best_params)
        clf.fit(X_train, y_train)
    elif args.model_config['model'] == 'AUPRCThreshold+LogisticRegression':
        # Make pipeline
        pipe = Pipeline([
            ('feature_selection', AUPRCThreshold()),
            ('classifier', SafeLogisticRegression())
        ])
        
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=pipe, 
            param_grid=args.model_config['param_grid'],
            cv=3, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)
        if np.isnan(grid_search.best_score_):
            raise ValueError('grid_search.best_score_ is nan.')
            
        # Retrieve the model
        clf = grid_search.best_estimator_
    elif args.model_config['model'] == 'AUPRCThreshold+LGBMClassifier':
        # Make pipeline
        pipe = Pipeline([
            ('feature_selection', AUPRCThreshold()),
            ('classifier', LGBMClassifier())
        ])
        
        # Gridsearch
        grid_search = GridSearchCV(
            estimator=pipe, 
            param_grid=args.model_config['param_grid'],
            cv=3, n_jobs=-1,
            scoring=get_scorer(args.model_config['scoring'])
        )
        grid_search.fit(X_train, y_train)
        print('Best score: {:.3f}'.format(grid_search.best_score_), end='\t', flush=True)
        
        # Retrieve the model
        clf = grid_search.best_estimator_
    return grid_search, clf


def get_scorer(scoring):
    if scoring in SCORERS.keys():
        return SCORERS[scoring]
    elif scoring == 'calibrated_f1':
        return make_scorer(calibrated_f1)
    elif scoring == 'calibrated_f1_macro':
        return make_scorer(calibrated_f1_macro)
    elif scoring == 'auprc':
        return make_scorer(auprc)
    else:
        raise ValueError('Invalid scoring')
        
        
def calibrated_f1(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return np.nan
    return f1_score(y_true, y_pred, sample_weight=compute_sample_weight('balanced', y_true))


def calibrated_f1_macro(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return np.nan
    return f1_score(y_true, y_pred, average='macro', sample_weight=compute_sample_weight('balanced', y_true))


def auprc(y_true, y_pred):
    return average_precision_score(y_true, y_pred, sample_weight=compute_sample_weight('balanced', y_true))


def calculate_max_features(X, y, threshold):
    n_features = X.shape[1]

    # calculate auprc for each feature
    auprc = np.zeros((n_features, ))
    for i in range(n_features):
        precision, recall, _ = precision_recall_curve(
            y, X[:, i], sample_weight=compute_sample_weight('balanced', y))
        auprc[i] = auc(recall, precision)

    return np.sum(auprc > threshold)


# class BinomialBayesMixedGLM:
#     def __process_data(self, X, y, subjects):
#         data = dict()

#         # Add features
#         n_samples, n_features = X.shape
#         for i in range(n_features):
#             data['x{:d}'.format(i+1)] = X[:, i]

#         # Add label
#         data['y'] = y

#         # Add groups
#         data['group'] = subjects

#         return pd.DataFrame(data)
    
#     def fit(self, X, y, subjects):
#         # Prepare data
#         data = self.__process_data(X, y, subjects)
        
#         # Create formula
#         formula = 'y ~ '
#         for col in data.columns:
#             if col.startswith('x'):
#                 formula += '{:s} + '.format(col)
#         formula = formula[:-2]

#         # Create model
#         model = sms.BinomialBayesMixedGLM.from_formula(
#             formula, {'group': '0 + C(group)'}, data
#         )

#         # Fit the model
#         self.result = model.fit_vb()
        
#         return self
    
#     def predict(self, X, subjects=None):
#         probability = self.predict_proba(X, subjects)
    
#         predictions = probability.copy()
#         predictions[probability > 0.5] = 1
#         predictions[probability <= 0.5] = 0
#         return predictions
    
#     def predict_proba(self, X, subjects=None):
#         # Extract fixed effects coefficients
#         fixed_effects = self.result.fe_mean

#         # Extract random effects
#         random_effects = self.result.random_effects().to_dict()

#         # Prepare X for multiplication
#         n_samples, n_features = X.shape
#         X = np.hstack((X, np.zeros((n_samples, 1))))

#         # Predict probability
#         log_odds = np.matmul(X, fixed_effects.reshape(-1, 1))
#         if subjects is not None:
#             for idx, subj in enumerate(subjects):
#                 log_odds[idx] += random_effects['Mean']['C(group)[{:d}]'.format(subj)]
#         probability = 1 / (1 + np.exp(-log_odds))
        
#         return probability