import re
import datetime
import numpy as np


def parse_index_value(index, value):
    # update index
    data_attr = index

    # update value
    if data_attr == 'workhr':
        if isinstance(value, str):
            value = re.findall(r'\d+', value)
            if value:
                value = np.mean(list(map(float, value)))
            else:
                value = np.nan
        elif isinstance(value, datetime.datetime):
            value = np.nan
    elif data_attr == 'precereb_number':
        if np.isnan(value):
            value = 0
    label = value
    return data_attr, label


def check_ischemic(data):
    '''
    Check if fields related to ischemic make sense
    '''
    if data.admit_stroke_dx___2 == 1:
        ischemic___1 = data.ischemic___1
        ischemic___2 = data.ischemic___2
        mask = ~(ischemic___1 & ischemic___2)

        data.ischemic___1 = ischemic___1 & mask
        data.ischemic___2 = ischemic___2 & mask
        data.ischemic___3 = ischemic___1 & ischemic___2
        data.ischemic___4 = ~(ischemic___1 | ischemic___2) & 1
    else:
        data.ischemic___1 = 0
        data.ischemic___2 = 0
        data.ischemic___3 = 0
        data.ischemic___4 = 0
    
        
def check_hemorrhagic(data):
    '''
    Check if fields related to hemorrhagic make sense
    '''
    if data.admit_stroke_dx___2 == 3:
        hemorrhagic_side___1 = data.hemorrhagic_side___1
        hemorrhagic_side___2 = data.hemorrhagic_side___2
        mask = ~(hemorrhagic_side___1 & hemorrhagic_side___2)

        data.hemorrhagic_side___1 = hemorrhagic_side___1 & mask
        data.hemorrhagic_side___2 = hemorrhagic_side___2 & mask
        data.hemorrhagic_side___3 = hemorrhagic_side___1 & hemorrhagic_side___2
        data.hemorrhagic_side___4 = ~(hemorrhagic_side___1 | hemorrhagic_side___2) & 1
    else:
        data.hemorrhagic_side___1 = 0
        data.hemorrhagic_side___2 = 0
        data.hemorrhagic_side___3 = 0
        data.hemorrhagic_side___4 = 0

        data.hemorrhagic_type___1 = 0
        data.hemorrhagic_type___2 = 0
        data.hemorrhagic_type___3 = 0


