import re
import datetime
import numpy as np
from enum import IntEnum
from collections import defaultdict


class HoursSleepLabel(IntEnum):
    NULL = 0
    HOURS_SLEEP_0_3 = 1
    HOURS_SLEEP_4_6 = 2
    HOURS_SLEEP_7_8 = 3
    HOURS_SLEEP_9_10 = 4
    HOURS_SLEEP_10_PLUS = 5
    
    @staticmethod
    def resolve(label):
        if label == '0-3':
            return HoursSleepLabel.HOURS_SLEEP_0_3
        elif label == datetime.datetime(2022, 4, 6, 0, 0):
            return HoursSleepLabel.HOURS_SLEEP_4_6
        elif label == datetime.datetime(2022, 7, 8, 0, 0):
            return HoursSleepLabel.HOURS_SLEEP_7_8
        elif label == datetime.datetime(2022, 9, 10, 0, 0):
            return HoursSleepLabel.HOURS_SLEEP_9_10
        elif label == '10+':
            return HoursSleepLabel.HOURS_SLEEP_10_PLUS
        elif np.isnan(label):
            return HoursSleepLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')
        
        
class FiveScaleLabel(IntEnum):
    NULL = 0
    NOT_AT_ALL = 1
    A_LITTLE_BIT = 2
    SOMEWHAT = 3
    QUITE_A_BIT = 4
    VERY_MUCH = 5

    @staticmethod
    def resolve(label):
        if label == 'Not at all':
            return FiveScaleLabel.NOT_AT_ALL
        elif label == 'A little bit':
            return FiveScaleLabel.A_LITTLE_BIT
        elif label == 'Somewhat':
            return FiveScaleLabel.SOMEWHAT
        elif label == 'Quite a bit':
            return FiveScaleLabel.QUITE_A_BIT
        elif label == 'Very much':
            return FiveScaleLabel.VERY_MUCH
        elif np.isnan(label):
            return FiveScaleLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')

    @staticmethod
    def retrieve(label):
        if label == 0:
            return 'NULL'
        elif label == 1:
            return 'NOT_AT_ALL'
        elif label == 2:
            return 'A_LITTLE_BIT'
        elif label == 3:
            return 'SOMEWHAT'
        elif label == 4:
            return 'QUITE A BIT'
        elif label == 5:
            return 'VERY MUCH'
        else:
            raise ValueError('Did not recognize the label.')
        

class YesNoLabel(IntEnum):
    NULL = 0
    NO = 1
    YES = 2
    OTHER = 3
    
    @staticmethod
    def resolve(label):
        if label == 'No':
            return YesNoLabel.NO
        elif label == 'Yes':
            return YesNoLabel.YES
        elif label == 'I don\'t need to take any medication': # difference between this and no?
            return YesNoLabel.OTHER
        elif label == 'I don\'t have any prescribed home exercise program or homework': # difference between this and no?
            return YesNoLabel.OTHER
        elif np.isnan(label):
            return HoursSleepLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')
        

class WhereAreYouLabel(IntEnum):
    NULL = 0
    AT_MY_HOME = 1
    AT_HOME_FAMILY = 2
    AT_HOME_FRIENDS = 3
    AT_WORK = 4
    AT_OUTPATIENT_MEDICAL_VISIT = 5
    IN_HOSPITAL = 6
    AT_RESTAURANT = 7
    AT_COMMUNITY_CENTER = 8
    IN_PUBLIC_STORE = 9
    IN_A_VEHICLE = 10
    OUTSIDE_WALKING = 11
    IN_CLASS = 12
    INSIDE_OTHER = 13
    OUTSIDE_OTHER = 14

    @staticmethod
    def resolve(label):
        if label == 'At my home':
            return WhereAreYouLabel.AT_MY_HOME
        elif label == 'At the home of family member(s)':
            return WhereAreYouLabel.AT_HOME_FAMILY
        elif label == 'At the home of friend(s)':
            return WhereAreYouLabel.AT_HOME_FRIENDS
        elif label == 'At work':
            return WhereAreYouLabel.AT_WORK
        elif label == 'At outpatient medical visit':
            return WhereAreYouLabel.AT_OUTPATIENT_MEDICAL_VISIT
        elif label == 'In hospital':
            return WhereAreYouLabel.IN_HOSPITAL
        elif label == 'At restaurant':
            return WhereAreYouLabel.AT_RESTAURANT
        elif label == 'At community center':
            return WhereAreYouLabel.AT_COMMUNITY_CENTER
        elif label == 'In public business/store (e.g., post office, grocery store)':
            return WhereAreYouLabel.IN_PUBLIC_STORE
        elif label == 'In a vehicle':
            return WhereAreYouLabel.IN_A_VEHICLE
        elif label == 'Outside, walking':
            return WhereAreYouLabel.OUTSIDE_WALKING
        elif label == 'In class/school setting':
            return WhereAreYouLabel.IN_CLASS
        elif label == 'Inside, other':
            return WhereAreYouLabel.INSIDE_OTHER
        elif label == 'Outside, other':
            return WhereAreYouLabel.OUTSIDE_OTHER
        elif np.isnan(label):
            return WhereAreYouLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')
    
                   
class HowGetWhereAreLabel(IntEnum):
    NULL = 0
    WALKED = 1
    RIDE_FROM_OTHERS = 2
    BUS_OR_TRAIN = 3
    DRIVE_SELF = 4
    OTHER = 5
    
    @staticmethod
    def resolve(label):
        if label == 'Walked':
            return HowGetWhereAreLabel.WALKED
        elif label == 'Got a ride from someone':
            return HowGetWhereAreLabel.RIDE_FROM_OTHERS
        elif label == 'Took bus or train':
            return HowGetWhereAreLabel.BUS_OR_TRAIN
        elif label == 'Drove myself':
            return HowGetWhereAreLabel.DRIVE_SELF
        elif label == 'Other':
            return HowGetWhereAreLabel.OTHER
        else:
            return HowGetWhereAreLabel.NULL
    
    
class CheckedLabel(IntEnum):
    UNCHECKED = 0
    CHECKED = 1
    
    @staticmethod
    def resolve(label):
        if label == 'Unchecked':
            return CheckedLabel.UNCHECKED
        elif label == 'Checked':
            return CheckedLabel.CHECKED
        else:
            raise ValueError('Did not recognize the label.')
        

class TimeSocializedLabel(IntEnum):
    NULL = 0
    NO_INTERACTIONS = 1
    ONE_INTERACTION = 2
    TWO_INTERACTIONS = 3
    THREE_INTERACTIONS = 4
    FOUR_MORE_INTERACTIONS = 5
    
    @staticmethod
    def resolve(label):
        if label == '0 (no interactions)':
            return TimeSocializedLabel.NO_INTERACTIONS
        elif label == '1 interaction':
            return TimeSocializedLabel.ONE_INTERACTION
        elif label == '2 interactions':
            return TimeSocializedLabel.TWO_INTERACTIONS
        elif label == '3 interactions':
            return TimeSocializedLabel.THREE_INTERACTIONS
        elif label == '4 or more interactions':
            return TimeSocializedLabel.FOUR_MORE_INTERACTIONS
        elif np.isnan(label):
            return WhereAreYouLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')


class ActivityLabel(IntEnum):
    NULL = 0

    # ADL
    CHANING_CLOTHES = 1 # ADL
    EATING_OR_DRINKING = 2 # ADL
    SHOWERING = 3 # ADL

    # Cognitively stimulating
    ARTS_AND_CARFTS = 4 # cognitively_stimulating
    CARE_OF_OTHERS = 5 # cognitively_stimulating ？
    INTERNET = 6 # cognitively_stimulating
    MEDITATING = 7 # cognitively_stimulating
    PLAYING_A_MUSICAL_INSTRUMENT = 8 # cognitively_stimulating
    PRIVATE_RELIGIOUS_ACTIVITIES = 9 # cognitively_stimulating
    READING = 10 # cognitively_stimulating
    SCHOOL_WORK = 11 # cognitively_stimulating
    SOCIAL_MEDIAL = 12 # cognitively_stimulating
    WORKING_PAID = 13 # cognitively_stimulating
    WORKING_UNPAID = 14 # cognitively_stimulating
    OTHER_MENTALLY_STIMULATING_ACTIVITY = 15 # cognitively_stimulating

    # IADL
    BUDGETING = 16 # IADL
    DOING_CHORES = 17 # IADL
    GETTING_GAS = 18 # IADL ?
    LOOKING_FOR_A_JOB = 19 # IADL
    PREPARING_FOOD = 20 # IADL
    RIDING_IN_A_BUS = 21 # IADL
    SHOPPING = 22 # IADL
    
    # Passive leisure
    LISTENING_TO_MUSIC = 23 # passive_leisure
    NOTHING = 24 # passive_leisure
    PLAYING_WITH_PETS = 25 # passive_leisure ? 
    RESTING = 26 # passive_leisure
    SMOKING = 27 # passive_leisure
    WATCHING_TV = 28 # passive_leisure
    
    # Physical activity
    EXERCISING = 29 # physical_activity
    GARDENING = 30 # physical_activity
    HOUSEHOLD_PROJECTS = 31 # physical_activity ?
    TRAVELING = 32 # physical_activity ?
    VISITING_BEACH = 33 # physical_activity
    VISITING_MUSEUM = 34 # physical_activity ?
    OTHER_PHYSICAL_ACTIVITY = 35 # physical_activity

    # Social activity
    EATING_OR_DRINKING_OUT = 36 # social_activity
    ENTERTAINMENT = 37 # social_activity
    MEETING = 38 # social_activity
    PARTICIPATING_IN_SOCIAL_EVENT = 39 # social_activity
    PLAYING_WITH_CHILDREN = 40 # social_activity ?
    SOCIAL_INTERACTIONS = 41 # social_activity
    TALKING_ON_THE_PHONE = 42 # social_activity ?
    VISITING_BARBER = 43 # social_activity ?
    VISITING_FAMILY = 44 # social_activity
    VISITING_HEALTHCARE = 45 # social_activity ?    
    OTHER_SOCIAL_ACTIVITY = 46 # social_activity

    @staticmethod
    def resolve(label):
        # ADL
        if label == 'Changing clothes':
            return ActivityLabel.CHANING_CLOTHES
        elif label == 'Eating or drinking at home':
            return ActivityLabel.EATING_OR_DRINKING
        elif label == 'Showering or grooming':
            return ActivityLabel.SHOWERING
        
        # Cognitively stimulating
        elif label == 'Arts and crafts':
            return ActivityLabel.ARTS_AND_CARFTS
        elif label == 'Care of others' or label == 'Caring for others':
            return ActivityLabel.CARE_OF_OTHERS
        elif label == 'Internet/computer/tablet use':
            return ActivityLabel.INTERNET
        elif label == 'Meditating':
            return ActivityLabel.MEDITATING
        elif label == 'Playing a musical instrument':
            return ActivityLabel.PLAYING_A_MUSICAL_INSTRUMENT
        elif label == 'Private religious activities':
            return ActivityLabel.PRIVATE_RELIGIOUS_ACTIVITIES
        elif label == 'Reading, writing, or journaling':
            return ActivityLabel.READING
        elif label == 'Schoolwork':
            return ActivityLabel.SCHOOL_WORK
        elif label == 'Social media (e.g., Facebook, Twitter)' or label == 'Social media (Facebook, Twitter)':
            return ActivityLabel.SOCIAL_MEDIAL
        elif label == 'Working (paid)':
            return ActivityLabel.WORKING_PAID
        elif label == 'Working (unpaid) or volunteering':
            return ActivityLabel.WORKING_UNPAID
        elif label == 'Other mentally stimulating activity':
            return ActivityLabel.OTHER_MENTALLY_STIMULATING_ACTIVITY
        
        # IADL
        elif label == 'Budgeting or paying bills':
            return ActivityLabel.BUDGETING
        elif label == 'Doing household chores (e.g., laundry, cleaning)' or label == 'Doing laundry away from home':
            return ActivityLabel.DOING_CHORES
        elif label == 'Getting gas':
            return ActivityLabel.GETTING_GAS
        elif label == 'Looking for a job':
            return ActivityLabel.LOOKING_FOR_A_JOB
        elif label == 'Preparing food/cooking':
            return ActivityLabel.PREPARING_FOOD
        elif label == 'Riding in a bus, trolley, car, or van':
            return ActivityLabel.RIDING_IN_A_BUS
        elif label == 'Shopping online' or label == 'Shopping outside of the home':
            return ActivityLabel.SHOPPING
        
        # Passive leisure
        elif label == 'Listening to music/radio':
            return ActivityLabel.LISTENING_TO_MUSIC
        elif label == 'Nothing':
            return ActivityLabel.NOTHING
        elif label == 'Playing with pets':
            return ActivityLabel.PLAYING_WITH_PETS
        elif label == 'Resting':
            return ActivityLabel.RESTING
        elif label == 'Smoking':
            return ActivityLabel.SMOKING
        elif label == 'Watching TV':
            return ActivityLabel.WATCHING_TV
        
        # Physical activity
        elif label == 'Exercising':
            return ActivityLabel.EXERCISING
        elif label == 'Gardening':
            return ActivityLabel.GARDENING
        elif label == 'Household projects/car maintenance':
            return ActivityLabel.HOUSEHOLD_PROJECTS
        elif label == 'Traveling':
            return ActivityLabel.TRAVELING
        elif label == 'Visiting the beach, park, etc.':
            return ActivityLabel.VISITING_BEACH
        elif label == 'Visiting a museum, concert hall, etc.':
            return ActivityLabel.VISITING_MUSEUM
        elif label == 'Other physical activity':
            return ActivityLabel.OTHER_PHYSICAL_ACTIVITY
        
        # Social activity
        elif label == 'Eating or drinking out':
            return ActivityLabel.EATING_OR_DRINKING_OUT
        elif label == 'Entertainment (cinema, sports, etc.)':
            return ActivityLabel.ENTERTAINMENT
        elif label == 'Meeting (church, parent group, AA, etc.)':
            return ActivityLabel.MEETING
        elif label == 'Participating in a social event (party, wedding, etc.)':
            return ActivityLabel.PARTICIPATING_IN_SOCIAL_EVENT
        elif label == 'Playing with children':
            return ActivityLabel.PLAYING_WITH_CHILDREN
        elif label == 'Social interactions with someone':
            return ActivityLabel.SOCIAL_INTERACTIONS
        elif label == 'Talking on the phone/texting':
            return ActivityLabel.TALKING_ON_THE_PHONE
        elif label == 'Visiting the barber/hairdresser/nail salon':
            return ActivityLabel.VISITING_BARBER
        elif label == 'Visiting family or friends':
            return ActivityLabel.VISITING_FAMILY
        elif label == 'Visiting healthcare providers (doctors, nurses, therapists)':
            return ActivityLabel.VISITING_HEALTHCARE
        elif label == 'Other social activity':
            return ActivityLabel.OTHER_SOCIAL_ACTIVITY
        
        elif np.isnan(label):
            return ActivityLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')
        

class PainLabel(IntEnum):
    NULL = 0
    NO_PAIN = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    VERY_SEVERE = 5
    
    @staticmethod
    def resolve(label):
        if label == 'No pain':
            return PainLabel.NO_PAIN
        elif label == 'Mild':
            return PainLabel.MILD
        elif label == 'Moderate':
            return PainLabel.MODERATE
        elif label == 'Severe':
            return PainLabel.SEVERE
        elif label == 'Very Severe':
            return PainLabel.VERY_SEVERE
        elif np.isnan(label):
            return PainLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')


class EMASurveyStatusLabel(IntEnum):
    NULL = 0
    MISSED = 1
    SCHEDULED = 2
    INCOMPLETE = 3
    IN_PROGRESS = 4
    COMPLETED = 5

    @staticmethod
    def resolve(label):
        if label == 'Missed':
            return EMASurveyStatusLabel.MISSED
        elif label == 'Incomplete':
            return EMASurveyStatusLabel.INCOMPLETE
        elif label == 'Scheduled':
            return EMASurveyStatusLabel.SCHEDULED
        elif label == 'In Progress':
            return EMASurveyStatusLabel.IN_PROGRESS
        elif label == 'Completed':
            return EMASurveyStatusLabel.COMPLETED
        elif np.isnan(label):
            return EMASurveyStatusLabel.NULL
        else:
            raise ValueError('Did not recognize the label.')


class EMASurveyCompleteLabel(IntEnum):
    NULL = 0
    INCOMPLETE = 1
    UNVERIFIED = 2
    COMPLETE = 3
    
    @staticmethod
    def resolve(label):
        if label == 'Incomplete':
            return EMASurveyCompleteLabel.INCOMPLETE
        elif label == 'Unverified':
            return EMASurveyCompleteLabel.UNVERIFIED
        elif label == 'Complete':
            return EMASurveyCompleteLabel.COMPLETE
        else:
            raise ValueError('Did not recognize the label.')


def return_class_label(data_attr, value):
    if data_attr == 'hours_sleep':
        return HoursSleepLabel.resolve(value)
    elif data_attr in ['outside', 'plan_social', 'medication', 'home_program']:
        return YesNoLabel.resolve(value)
    elif data_attr == 'where_are_you':
        return WhereAreYouLabel.resolve(value)
    elif data_attr == 'how_get_where_are':
        return HowGetWhereAreLabel.resolve(value)
    # elif data_attr == [
    #     'goal',
    #     'environmental_barriers', 
    #     'who_is_with_you', 
    #     'why_not_engaged_social',
    #     'why_not_doing_anything',
    #     'redo', 'clinician_help']:
    #     return CheckedLabel.resolve(value)
    elif '___' in data_attr:
        return CheckedLabel.resolve(value)   
    elif data_attr == 'times_socialized':
        return TimeSocializedLabel.resolve(value) 
    elif data_attr == 'what_doing_home':
        return ActivityLabel.resolve(value) 
    elif data_attr == 'doing_not_home':
        return ActivityLabel.resolve(value)    
    elif data_attr == 'pain':
        return PainLabel.resolve(value)
    elif data_attr == 'ema_survey_complete':
        return EMASurveyCompleteLabel.resolve(value)
    elif data_attr == 'ema_survey_status':
        return EMASurveyStatusLabel.resolve(value)
    elif data_attr in [
        'falling_asleep', 'confidence_goal',
        'isolated', 'stress', 'tired',
        'depressed', 'worthless', 'concentrating',
        'learning_new', 'anxiety', 'mindfullness',
        'little_interest', 'appetite', 
        'slow_restless', 'cheerful', 
        'success_goal', 'satisfied_day', 
        'lookward_social', 'feasibility',
        'appropriateness', 'acceptability',
        'acceptability2']:
        return FiveScaleLabel.resolve(value)
    else:
        return value
    
    
def parse_index_value(index, value):
    # parse index
    pattern = r'^(.*?)_[123](.*)$'
    res = re.match(pattern, index)
    res1, res2 = res[1], res[2]

    # # update index
    # if not res2.startswith('___'):
    #     data_attr = res1 + res2
    # else:
    #     data_attr = res1
          
    # # update value
    # if res2.startswith('___'):
    #     if CheckedLabel.resolve(value) == 0:
    #         label = []
    #     else:
    #         label = [int(res2.split('___')[1])]
    # else:
    #     label = return_class_label(data_attr, value)
    # return data_attr, label

    # update index
    data_attr = res1 + res2
          
    # update value
    label = return_class_label(data_attr, value)
    return data_attr, label


def check_completeness(data):
    # checkbox questions
    # one of the choices must be selected
    checkbox_questions = defaultdict(list)

    # slider questions
    # the range must be in [1, 7]
    slider_attrs = [
        'confidence_social', 'satisfaction_social', 
        'success_social', 'help', 'perform_activity', 
        'satisfied_activity', 'engaged_activity', 
        'effort_social', 'pleasure_social']
    slider_valid = True
    
    # the rest questions
    is_valid = []

    for data_attr in dir(data):
        if data_attr.startswith('__'):
            continue
        
        if data_attr.startswith('ema'):
            continue

        if '___' in data_attr:
            # checkbox questions
            if getattr(data, data_attr) == 1:
                name = data_attr.split('___')
                checkbox_questions[name[0]].append(int(name[1]))
        elif data_attr in slider_attrs:
            # attributes with slider values should be in [1, 7]
            if not np.isnan(getattr(data, data_attr)):
                slider_valid &= (1 <= getattr(data, data_attr) <= 7)
        else:
            # check if the rest attributes are valid or not
            if isinstance(getattr(data, data_attr), float) and not np.isnan(getattr(data, data_attr)):
                is_valid.append(True)
            elif isinstance(getattr(data, data_attr), IntEnum) and getattr(data, data_attr) != 0:
                is_valid.append(True)
            else:
                is_valid.append(False)

    # checkbox questions
    for key in checkbox_questions:
        is_valid.append(len(checkbox_questions[key]) > 0) 
    return np.mean(is_valid) > 0.5 and slider_valid and data.isolated != 0


def preprocess_survey_data(data):
    # check goal attribute
    if 'goal___1' in dir(data):
        data = __preprocess_checkbox_questions(data, 'goal')

    # check environmental_barriers
    data = __preprocess_checkbox_questions(data, 'environmental_barriers')

    # check where_are_you and how_get_where_are
    __preprocess_where_are_you(data)

    # # check who_is_with_you
    # data = __preprocess_checkbox_questions(data, 'who_is_with_you')

    # check who_is_with_you and social activity
    data = __preprocess_who_is_with_you(data)

    # check who_is_with_you and why_not_engaged_social

    # check what_doing_home and doing_not_home
    data = __preprocess_location_and_activity(data)

    # check secondary condition
    __preprocess_secondary_condition(data)

    return data


def __preprocess_checkbox_questions(data, index):
    values = []
    for data_attr in sorted(dir(data)):
        if data_attr.startswith(index) and getattr(data, data_attr) == 1:
            values.append(int(data_attr.split('___')[1]))
    if len(values) == 0:
        setattr(data, '{index}___1', 1)
    elif len(values) > 1 and 1 in values:
        setattr(data, '{index}___1', 0)
    return data


def __preprocess_where_are_you(data):
    if data.where_are_you == WhereAreYouLabel.AT_MY_HOME and data.how_get_where_are != HowGetWhereAreLabel.NULL:
        raise ValueError('at home and how_get_where_are is invalid.')
    elif data.where_are_you not in [0, 1] and data.how_get_where_are == HowGetWhereAreLabel.NULL:
        raise ValueError('not at home and how_get_where_are is invalid')


def __preprocess_who_is_with_you(data):
    if data.who_is_with_you___1 == 0 and data.who_is_with_you___6 == 0:
        if np.isnan(data.confidence_social):
            raise ValueError('with people and confidence_social is nan.')
        if np.isnan(data.satisfaction_social):
            raise ValueError('with people and satisfaction_social is nan.')
        if np.isnan(data.success_social):
            raise ValueError('with people and success_social is nan.')
    else:
        if np.isnan(data.confidence_social):
            data.confidence_social = 0
        if np.isnan(data.satisfaction_social):
            data.satisfaction_social = 0
        if np.isnan(data.success_social):
            data.success_social = 0
    return data
        
    
def __preprocess_location_and_activity(data):
    where_are_you = data.where_are_you
    what_doing_home = data.what_doing_home
    doing_not_home = data.doing_not_home

    # check if location and activity match
    if (what_doing_home == ActivityLabel.NULL and
        doing_not_home == ActivityLabel.NULL):
        # both activities are null, raise error
        raise ValueError('what_doing_home is NULL and doing_not_home is NULL')
    elif (what_doing_home != ActivityLabel.NULL and
          doing_not_home != ActivityLabel.NULL):
        # both activities are not null
        # change value based on location
        if where_are_you == WhereAreYouLabel.AT_MY_HOME:
            data.doing_not_home = ActivityLabel.NULL
        else:
            data.what_doing_home = ActivityLabel.NULL
    elif (what_doing_home != ActivityLabel.NULL and 
          doing_not_home == ActivityLabel.NULL):
        if where_are_you != WhereAreYouLabel.AT_MY_HOME:
            # raise ValueError('what_doing_home is not NULL and where_are_you not home')
            data.what_doing_home = ActivityLabel.NULL
            data.doing_not_home = getattr(ActivityLabel, what_doing_home.name)
    elif (what_doing_home == ActivityLabel.NULL and
          doing_not_home != ActivityLabel.NULL):
        if where_are_you == WhereAreYouLabel.AT_MY_HOME:
            # raise ValueError('doing_not_home is not NULL and where_are_you at home')
            data.what_doing_home = getattr(ActivityLabel, doing_not_home.name)
            data.doing_not_home = ActivityLabel.NULL

    # check activity and performance
    if what_doing_home not in [ActivityLabel.NULL, ActivityLabel.RESTING, ActivityLabel.NOTHING] and \
       doing_not_home not in [ActivityLabel.NULL, ActivityLabel.RESTING, ActivityLabel.NOTHING]:
        if np.isnan(data.help):
            raise ValueError('doing activity not help is invalid')
        if np.isnan(data.perform_activity):
            raise ValueError('doing activity not perform_activity is invalid') 
        if np.isnan(data.satisfied_activity):
            raise ValueError('doing activity not satisfied_activity is invalid') 
        if np.isnan(data.engaged_activity):
            raise ValueError('doing activity not engaged_activity is invalid')
    elif what_doing_home in [ActivityLabel.RESTING, ActivityLabel.NOTHING] or \
         doing_not_home in [ActivityLabel.RESTING, ActivityLabel.NOTHING]:
        if np.isnan(data.help):
            data.help = 0
        if np.isnan(data.perform_activity):
            data.perform_activity = 0
        if np.isnan(data.satisfied_activity):
            data.satisfied_activity = 0
        if np.isnan(data.engaged_activity):
            data.engaged_activity = 0
    return data


def __preprocess_secondary_condition(data):
    secondary_condition = [
        'pain', 'isolated', 'stress', 
        'tired', 'depressed', 'worthless', 
        'concentrating', 'learning_new', 
        'anxiety', 'mindfullness',
        'little_interest', 'appetite', 
        'slow_restless', 'cheerful'
    ]
    for data_attr in secondary_condition:
        if getattr(data, data_attr) == 0:
            raise ValueError(r'{data_attr} is invalid')
