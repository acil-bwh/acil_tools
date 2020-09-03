import numpy as np
from os.path import join, realpath, dirname
import pandas as pd
import pdb

def hankinson(spiro_type, race, gender, age, height,
              co_delta=np.zeros(4)):
    """Computes the predicted FEV1 or FVC based on the equations given in the
    reference below (Hankinson et al).

    Parameters
    ----------
    spiro_type : string
        Either 'fev1' or 'fvc'

    race : string
        Either 'caucasian', 'african_american', or 'mexican_american'

    gender : string
        Either 'male' or 'female'

    age : float, or vector
        Person's age in years

    height : float, or vector
        Person's height in cm

    co_delta : vector of floats, optional
        If specified, this vector will be added to the demographic-specific
        coefficient vector. This can be useful when generating synthetic data.
        Ordering of the coefficient vector is: intercept, age, age^2, height^2.

    Returns
    -------
    value : float or vector
        FEV1 or FVC depending on spiro_type

    References
    ----------
    [1] Hankinson, John L., John R. Odencrantz, and Kathleen B. Fedan.
    "Spirometric reference values from a sample of the general US population."
    American journal of respiratory and critical care medicine 159,
    no. 1 (1999): 179-187.
    """
    if spiro_type != 'fev1' and spiro_type != 'fvc':
        raise ValueError('Unrecognized value')
    if np.sum(height <= 0) > 0:
            raise ValueError('Impossible height value')
    if gender != 'male' and gender != 'female':
        raise ValueError('Unrecognized gender')
    if race != 'caucasian' and race != 'african_american' and \
      race != 'mexican_american':
        raise ValueError('Unrecognized race')

    if isinstance(age, np.ndarray):
        N = age.shape[0]
    else:
        N = 1
        age = np.array([age])

    if isinstance(height, np.ndarray):
        vals = np.array([np.array([1]*N), age, age**2, height**2])
    else:
        vals = np.array([np.array([1]*N), age, age**2,
                         np.array([height**2]*N)])

    cos_lt = np.zeros(4)
    cos_gt = np.zeros(4)

    if gender == 'male':
        ids_lt = age < 20
        ids_gt = age >= 20
    else:
        ids_lt = age < 18
        ids_gt = age >= 18

    # Caucasian male < 20 years old
    if race == 'caucasian' and gender == 'male':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.7453, -0.04106, 0.004477, 0.00014098])
        else:
            cos_lt = np.array([-0.2584, -0.20415, 0.010133, 0.00018642])

    # Caucasian male >= 20 years old
    if race == 'caucasian' and gender == 'male':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.5536, -0.01303, -0.000172, 0.00014098])
        else:
            cos_gt = np.array([-0.1933, 0.00064, -0.000269, 0.00018642])

    # African American male < 20 years old
    if race == 'african_american' and gender == 'male':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.7048, -0.05711, 0.004316, 0.00013194])
        else:
            cos_lt = np.array([-0.4971, -0.15497, 0.007701, 0.00016643])

    # African American male >= 20 years old
    if race == 'african_american' and gender == 'male':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.3411, -0.02309, 0, .00013194])
        else:
            cos_gt = np.array([-0.1517, -0.01821, 0, 0.00016643])

    # Mexican American male < 20 years old
    if race == 'mexican_american' and gender == 'male':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.8218, -0.04248, 0.004291, 0.00015104])
        else:
            cos_lt = np.array([-0.7571, -0.09520, 0.006619, 0.00017823])

    # Mexican American male >= 20 years old
    if race == 'mexican_american' and gender == 'male':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.6306, -0.02928, 0, 0.00015104])
        else:
            cos_gt = np.array([0.2376, -0.00891, -0.000182, 0.00017823])

    # Caucasian female < 18 years old
    if race == 'caucasian' and gender == 'female':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.8710, 0.06537, 0, 0.00011496])
        else:
            cos_lt = np.array([-1.2082, 0.05916, 0, 0.00014815])

    # Caucasian female >= 18 years old
    if race == 'caucasian' and gender == 'female':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.4333, -0.00361, -0.000194, 0.00011496])
        else:
            cos_gt = np.array([-0.3560, 0.01870, -0.000382, 0.00014815])

    # African American female < 18 years old
    if race == 'african_american' and gender == 'female':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.9630, 0.05799, 0, .00010846])
        else:
            cos_lt = np.array([-0.6166, -0.04687, 0.003602, 0.00013606])

    # African American female >= 18 years old
    if race == 'african_american' and gender == 'female':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.3433, -0.01283, -0.000097, 0.00010846])
        else:
            cos_gt = np.array([-0.3039, 0.00536, -0.000265, 0.00013606])

    # Mexican American female < 18 years old
    if race == 'mexican_american' and gender == 'female':
        if spiro_type == 'fev1':
            cos_lt = np.array([-0.9641, 0.06490, 0, 0.00012154])
        else:
            cos_lt = np.array([-1.2507, 0.07501, 0, 0.00014246])

    # Mexican American female >= 18 years old
    if race == 'mexican_american' and gender == 'female':
        if spiro_type == 'fev1':
            cos_gt = np.array([0.4529, -0.01178, -0.000113, 0.00012154])
        else:
            cos_gt = np.array([0.1210, 0.00307, -0.000237, 0.00014246])

    values = np.zeros(N)
    if np.sum(ids_lt) > 0:
        values[ids_lt] = np.dot(cos_lt + co_delta, vals[:, ids_lt])
    if np.sum(ids_gt) > 0:
        values[ids_gt] = np.dot(cos_gt + co_delta, vals[:, ids_gt])

    return values

def gli_2012(spiro_type, race, gender, age, height, measured=None):
    """Used the Global Lung Function Initiative (GLI) 2012 spirometry equations
    to compute predicted, lower-limit of normal, z-scores, and percent
    predicted values.

    Parameters:
    -----------
    spiro_type : string
        Either 'fev1', fvc', or 'fev1fvc'

    race : string
        Either 'caucasian', 'african_american', 'ne_asian', 'se_asian', 'other'

    gender : string
        Either 'male' or 'female'

    age : float, or vector
        Person's age in years

    height : float, or vector
        Person's height in cm

    measured : float, optional

    Returns:
    --------
    ret : array
        Returned vectory has either 2 (if 'measured' is None), or 4 elements. 
        The first two elements are the predicted value, and the lower limit of 
        normal. If measured values are provided, then the next two elements are 
        the z-score and the percent-predicted value.
    """
    spiro_types = ['FEV1FVC', 'FEV1', 'FEF2575', 'FEV075FVC',
                   'FEF75', 'FEV075', 'FVC']
    if spiro_type.upper() not in spiro_types:
        raise ValueError('Unrecognized spirometry type')
    
    race_map = {'caucasian': 0,
                'african_american': 0,
                'ne_asian': 0,
                'se_asian': 0,
                'other': 0}
    if race not in race_map.keys():
        raise ValueError('Unrecognized racial group')

    gender_map = {'male': 1, 'female': 2}    
    race_map[race] = 1.

    if isinstance(age, np.ndarray):
        N = age.shape[0]
    else:
        N = 1
        age = np.array([age])
    
    #--------------------------------------------------------------------------
    # Read in LUT and retrieve vecs a, p, and q
    #--------------------------------------------------------------------------
    dir_path = dirname(realpath(__file__))
    lut_file_name = join(dir_path, 'resources/gli_2012_lookup_table.csv')
    
    df_lut = pd.read_csv(lut_file_name)
    gb = df_lut.groupby(['f', 'sex'])

    df_lut_group = gb.get_group((spiro_type.upper(), gender_map[gender]))
    a_vec = df_lut_group[['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']].\
        values[0, :]
    p_vec = df_lut_group[['p0', 'p1', 'p2', 'p3', 'p4', 'p5']].values[0, :]
    q_vec = df_lut_group[['q0', 'q1']].values[0, :]

    #--------------------------------------------------------------------------
    # Compute mspline, sspline, and lspline
    #--------------------------------------------------------------------------
    mspline = np.zeros(N)
    sspline = np.zeros(N)
    lspline = np.zeros(N)
    for (inc, a) in enumerate(age):
        for (i, aa) in enumerate(df_lut_group.agebound.values):
            if aa > a:
                i_max = i
                break

        age_high = df_lut_group.agebound.values[i_max]
        age_low = df_lut_group.agebound.values[i_max-1]
        
        m_high = df_lut_group.m0.values[i_max]
        m_low = df_lut_group.m0.values[i_max-1]
        mspline[inc] = m_low + (m_high-m_low)*(a-age_low)/(age_high-age_low)

        s_high = df_lut_group.s0.values[i_max]
        s_low = df_lut_group.s0.values[i_max-1]
        sspline[inc] = s_low + (s_high-s_low)*(a-age_low)/(age_high-age_low)

        l_high = df_lut_group.l0.values[i_max]
        l_low = df_lut_group.l0.values[i_max-1]
        lspline[inc] = l_low + (l_high-l_low)*(a-age_low)/(age_high-age_low)

    #--------------------------------------------------------------------------
    # Compute L, S, M
    #--------------------------------------------------------------------------=
    tmp_arr = np.ones([N, 7])
    tmp_arr[:, 1] *= np.log(height)
    tmp_arr[:, 2] = np.log(age)
    tmp_arr[:, 3] *= race_map['african_american']
    tmp_arr[:, 4] *= race_map['ne_asian']
    tmp_arr[:, 5] *= race_map['se_asian']
    tmp_arr[:, 6] *= race_map['other']    

    L = q_vec[0] + q_vec[1]*np.log(age) + lspline
    M = np.exp(np.dot(a_vec, tmp_arr.T) + mspline)

    tmp_arr = np.ones([N, 6])
    tmp_arr[:, 1] *= np.log(age)
    tmp_arr[:, 2] *= race_map['african_american']
    tmp_arr[:, 3] *= race_map['ne_asian']
    tmp_arr[:, 4] *= race_map['se_asian']
    tmp_arr[:, 5] *= race_map['other']

    S = np.exp(np.dot(p_vec, tmp_arr.T) + sspline)

    #--------------------------------------------------------------------------
    # Compile values and return
    #--------------------------------------------------------------------------
    pred_vals = np.atleast_2d(M).T
    llns = np.atleast_2d(np.exp(np.log(M) + np.log(1.-1.645*L*S)/L)).T

    if measured is not None:
        z_score = np.atleast_2d((((measured/M)**L)-1)/(L*S)).T
        perc_pred = np.atleast_2d(100.*measured/M).T
        if N==1:
            return np.hstack([pred_vals, llns, z_score, perc_pred])[0, :]
        else:
            return np.hstack([pred_vals, llns, z_score, perc_pred])
    else:
        if N==1:
            return np.hstack([pred_vals, llns])[0, :]
        else:
            return np.hstack([pred_vals, llns])            

