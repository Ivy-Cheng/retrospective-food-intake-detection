#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys 
import pandas as pd
import numpy as np
import datetime
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# ## Preparation

# In[4]:


#----------------------------------------------------------------------------
# read files about a patient in Ohio dataset
#----------------------------------------------------------------------------
def read_ohio_data(patient_id, features, path='../OhioT1DM/processed'):

    pid = str(patient_id)
    files = os.listdir(path)
    result = {}
    for feat in features:
        matched = [f for f in files if f'_{pid}_' in f and feat in f]
        dfs = []
        for fname in matched:
            full = os.path.join(path, fname)
            dfs.append(pd.read_csv(full))
        if dfs:
            result[f"{feat}_data"] = pd.concat(dfs, ignore_index=True)
        else:
            result[f"{feat}_data"] = pd.DataFrame()
    return result


# In[5]:


#----------------------------------------------------------------------------
# read files about a patient in UoM dataset
#----------------------------------------------------------------------------
def read_uom_data(patient_id, features, path='../UoM/new_Processed'):

    pid = str(patient_id)
    files = os.listdir(path)
    result = {}
    for feat in features:
        matched = [f for f in files if f'_{pid}_' in f and feat in f]
        dfs = []
        for fname in matched:
            full = os.path.join(path, fname)
            dfs.append(pd.read_csv(full))
        if dfs:
            result[f"{feat}_data"] = pd.concat(dfs, ignore_index=True)
        else:
            result[f"{feat}_data"] = pd.DataFrame()
    return result


# In[6]:


#----------------------------------------------------------------------------
# read files about a patient in Diatrend dataset
#----------------------------------------------------------------------------
def read_diatrend_data(patient_id, features, path='../Diatrend/Processed'):

    pid = str(patient_id)
    files = os.listdir(path)
    result = {}
    for feat in features:
        # 精确匹配患者ID：使用下划线作为分隔符确保精确匹配
        matched = [f for f in files if f'_{pid}_' in f and feat in f]  
        dfs = []
        for fname in matched:
            full = os.path.join(path, fname)
            dfs.append(pd.read_csv(full))
        if dfs:
            result[f"{feat}_data"] = pd.concat(dfs, ignore_index=True)
        else:
            result[f"{feat}_data"] = pd.DataFrame()
    return result


# In[7]:


# Set 'ts' as index for dataframe
def set_ts_as_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
    
    return df


# ## Add trend indicators
# Include Medtronic, Dexcom indicators<br>
# '↑' : 'rising'<br>
# '↑↑' : 'rising quickly'<br>
# '↑↑↑' : 'rising sharply'<br>
# '-' : 'steady'<br>
# '↓' : 'falling'<br>
# '↓↓' : 'falling quickly'<br>
# '↓↓↓' : 'falling sharply'<br>

# In[8]:


#--------------------------------------------------------------------------------
# Medtronic Trend Indicators
#--------------------------------------------------------------------------------
def medtronic_trends_20min(df, glucose_col='value'):
    # shift to 20 min ago
    df['G_20m_ago'] = df[glucose_col].shift(freq='20min')
    df['delta_20m'] = df[glucose_col] - df['G_20m_ago']

    # buckets: 640G
    # conditions = [
    #     df['delta_20m'].between(1, 2, inclusive='right'),
    #     df['delta_20m'].between(2, 3, inclusive='right'),
    #     df['delta_20m'] > 3,
    #     df['delta_20m'].between(-2, -1, inclusive='left'),
    #     df['delta_20m'].between(-3, -2, inclusive='left'),
    #     df['delta_20m'] < -3,
    #   ]

    # version 2:780G
    conditions = [
        df['delta_20m'].between(1.11, 2.22, inclusive='right'),
        df['delta_20m'].between(2.22, 3.33, inclusive='right'),
        df['delta_20m'] > 3.4,
        df['delta_20m'].between(-2.22, -1.11, inclusive='left'),
        df['delta_20m'].between(-3.33, -2.22, inclusive='left'),
        df['delta_20m'] < -3.4,
    ]
    choices = [
        'rising',
        'rising quickly',
        'rising sharply',
        'falling',
        'falling quickly',
        'falling sharply',
    ]
    df['medtronic_trend_20m'] = np.select(conditions, choices, default='-')
    df.loc[df['G_20m_ago'].isna(), 'medtronic_trend_20m'] = np.nan

    return df


# In[9]:


#--------------------------------------------------------------------------------
# Dexcom G7 Trend Indicators (Table I)
#--------------------------------------------------------------------------------
def dexcom_trends_15min(df, glucose_col='value'):
    # Glucose change over 15 minutes; input glucose is in mmol/L.
    df['G_15m_ago'] = df[glucose_col].shift(freq='15min')
    df['delta_15m'] = df[glucose_col] - df['G_15m_ago']

    # Exact mg/dL-to-mmol/L conversion underlying Table I's displayed values.
    conditions = [
        df['delta_15m'].between(15/18, 30/18, inclusive='right'),
        df['delta_15m'].between(30/18, 45/18, inclusive='right'),
        df['delta_15m'] > 45/18,
        df['delta_15m'].between(-30/18, -15/18, inclusive='left'),
        df['delta_15m'].between(-45/18, -30/18, inclusive='left'),
        df['delta_15m'] < -45/18,
    ]
    choices = [
        'rising',
        'rising quickly',
        'rising sharply',
        'falling',
        'falling quickly',
        'falling sharply',
    ]
    df['dexcom_trend_15m'] = np.select(conditions, choices, default='-')
    df.loc[df['G_15m_ago'].isna(), 'dexcom_trend_15m'] = np.nan
    return df


# In[11]:


def trend_indicators(df, glucose_col='value'):
    # ensure datetime index
    df = df.copy()
    df = set_ts_as_index(df)
    
    df = medtronic_trends_20min(df, glucose_col)
    df = dexcom_trends_15min(df, glucose_col)

    return df


# In[12]:


# -------------------------------------------------------------------------------------
# quantify the trend indicators into categorical values
# -------------------------------------------------------------------------------------
def full_trend_indicators(glucose_data):
    example = trend_indicators(glucose_data)

    quant_map = {
        'rising':            1,
        'rising quickly':    2,
        'rising sharply':    3,
        'falling':          -1,
        'falling quickly':  -2,
        'falling sharply':  -3,
        '-': 0
    }

    # if you want to keep the original text column, write to a new one:
    example['medtronic_20m'] = example['medtronic_trend_20m'].map(quant_map)
    example['dexcom_15m'] = example['dexcom_trend_15m'].map(quant_map)

    return example


# ## Further analysis

# In[15]:


def compute_auc(df, trend_col='medtronic_trend_20m_num'):
    """
    calculate continuous sections' AUC, and show whenther it is above 0 or not.
    return: ['start','end','length','auc','sign']
    sign: 1(positive), -1(negative), 0, Nan
    """
    df2 = df.copy()
    df2['sign'] = np.sign(df2[trend_col])
    df2['group'] = (df2['sign'] != df2['sign'].shift()).cumsum()
    
    records = []
    for _, grp in df2.groupby('group'):
        start = grp.index[0]
        end = grp.index[-1]
        length = len(grp)
        sign0 = grp['sign'].iloc[0]

        if pd.isna(sign0):
            auc = np.nan
        elif sign0 == 0:
            auc = 0.0
        else:
            # AUC
            y = grp[trend_col].values
            auc = ((y[:-1] + y[1:]) * 0.5).sum()
        
        sign_val = int(sign0) if not pd.isna(sign0) else np.nan
        records.append({
            'start': start,
            'end':   end,
            'length': length,
            'auc':    auc,
            'sign':   sign_val
        })
    
    return pd.DataFrame.from_records(records)


# In[16]:


def find_quantile(df, col, quantile):
    df['sign'] = df['sign'].astype(str)
    filtered_data = df[df['sign'] == '1.0'][col]
    quantile_value = filtered_data.quantile(quantile)
    return quantile_value



# ## Method pipelines

# ### Simple heuristic method
# 

# In[17]:


from pipelines.simple_heuristic_utils import simple_heuristic

def simple_method(glucose_data):
    # day_list = pd.date_range(start='2023-11-07', end='2023-11-07').date.tolist()

    day_list = pd.to_datetime(glucose_data["ts"]).dt.date.unique().tolist()
    simple_result= simple_heuristic(glucose_data,day_list)
    return simple_result


# ### Trend indicators method

# In[18]:


def check_trend_in_interval(row, merged_df):
    start_time = row['start']
    end_time = row['end']
    
    mask = (merged_df.index >= start_time) & (merged_df.index <= end_time)
    interval_data = merged_df.loc[mask, 'medtronic_20m']
    
    # count the number of values greater than 1
    count = (interval_data > 1).sum()
    

    # return True if there are more than 3 minutes greater than 1
    return count > 3


# In[19]:


def trend_method(merged):
    res = compute_auc(merged, 'medtronic_20m')
    res['has_multiple_trends'] = res.apply(lambda row: check_trend_in_interval(row, merged), axis=1)
    auc_q1 = find_quantile(res, 'auc', 0.25)
    print(auc_q1)
    length_q1 = find_quantile(res, 'length', 0.25)
    print(length_q1)
    mask = (res['auc'] >= auc_q1) & (res['length'] >= length_q1) & (res['has_multiple_trends']==True)
    trend_result_ts = res[mask]['start'].tolist()

    return trend_result_ts


# ### Physio-aided methods

# In[20]:


#---------Horvorka+UKF method-------------
import pipelines.physio_kf as pk

def physio_kf_method(merged):
    out = pk.physio_kf_method(
        merged,
        model='bergman',
        filter_type='ukf',
        impl='package',
        value_col='value',
        iob_col='iob',
        dose_col='dose',
        insulin_input='iob_activity',
        # threshold decisions: 
        # default values are 2/18, 100/18, 1, 10
        ra_thresh_mmol_per_min= 3/18,
        g_thresh_mmol= 4,
        gap_allow_min= 1,
        min_contiguous_min= 10
    )
    return out


# ## Datasets pipelines

# In[21]:


# import glucose_visualization
# from glucose_visualization import create_interactive_trend_figure


# ### Other support functions

# In[22]:


# 假设你的数据在 basal_data 中，列名为 'ts' 和 'value'
def convert_basal_to_min(basal_data):
    # in df basal_data, rename 'value' column name as 'basal'
    basal_data.rename(columns={'value': 'basal'}, inplace=True)
    # drop the 'insulin_kind' column
    basal_data.drop(columns=['insulin_kind'], inplace=True)
    basal_data.index = pd.to_datetime(basal_data.index)

    if basal_data.index.duplicated().any():
        print(f"发现 {basal_data.index.duplicated().sum()} 个重复时间戳，正在处理...")
        # 保留最后一个值
        basal_data = basal_data[~basal_data.index.duplicated(keep='last')]
    
    # 创建每分钟的时间范围
    start_time = basal_data.index.min().floor('H')  # 整点开始
    end_time = basal_data.index.max().ceil('H')     # 整点结束
    minute_range = pd.date_range(start_time, end_time, freq='1min')
    
    # 重采样到分钟级别，使用前向填充
    basal_resampled = basal_data.reindex(minute_range, method='ffill')
    basal_resampled.index.name = 'ts'
    
    return basal_resampled


# In[23]:


def remove_glucose_duplicates(glucose_data):
    """
    删除重复的时间戳行，在删除前计算相同时间戳的'value'列平均值
    
    参数:
        glucose_data: DataFrame包含'ts'和'value'列
    
    返回:
        DataFrame: 去重后的数据，相同时间戳的value已被平均值替换
    """
    # 创建副本避免修改原数据
    df = glucose_data.copy()
    
    print(f"原始数据行数: {len(df)}")
    
    # 检查重复情况
    duplicated_count = df['ts'].duplicated().sum()
    print(f"重复时间戳数量: {duplicated_count}")
    
    if duplicated_count > 0:
        # 按时间戳分组，计算value的平均值，保留其他列的第一个值
        df_grouped = df.groupby('ts').agg({
            'value': 'mean',  # 计算value的平均值
            'date': 'first',  # 保留第一个date
            'missing_flag': 'first'  # 保留第一个missing_flag
        }).reset_index()   
        return df_grouped
    else:
        print("没有发现重复的时间戳")
        return df


# In[24]:


def remove_bolus_duplicates(bolus_data):
# if the data is about bolus and the duplicate values are the same, keep the first one; 
# if the duplicate values are different, calculate the sum of the bolus      
    df = bolus_data.copy()
    print(f"原始数据行数: {len(df)}")
    # drop the rows where the dose is 0
    df = df[df['dose'] != 0]
    duplicated_count = df['ts'].duplicated().sum()
    print(f"重复时间戳数量: {duplicated_count}")
    if duplicated_count > 0:
        grp = df.groupby('ts', sort=True)

        nunq   = grp['dose'].nunique()
        sum_v  = grp['dose'].sum()
        mean_v = grp['dose'].mean()

        value = np.where((nunq == 1).values, mean_v.values, sum_v.values)
        df_revised = pd.DataFrame({'ts': nunq.index, 'dose': value})

        if 'date' in df.columns:
            df_revised['date'] = grp['date'].first().values
        if 'missing_flag' in df.columns:
            df_revised['missing_flag'] = grp['missing_flag'].first().values
    else:
        print("没有发现胰岛素注射数据重复的时间戳")
        df_revised = df

    return df_revised

def remove_meal_duplicates(meal_data):
    df = meal_data.copy()
    print(f"原始数据行数: {len(df)}")
    df = df[df['carbs'] != 0]
    duplicated_count = df['ts'].duplicated().sum()
    print(f"重复时间戳数量: {duplicated_count}")
    if duplicated_count > 0:
        df_revised = df.groupby('ts').agg({
            'type': 'first',
            'meal_tag': 'first',
            'carbs': 'sum',
            'protein': 'sum',
            'fat': 'sum',
            'fibre': 'sum'
        }).reset_index()
        return df_revised
    else:
        print("没有发现餐点数据重复的时间戳")
        return df


# ### Calculate IOB + Estimate CHO

# In[25]:


# --- 1. Activity/decay kernel ---------------------------------------------
def iob_fraction(t_minutes, duration_min=240):
    """
    Vectorised Walsh/Loop fraction of insulin still active after t_minutes.
    t_minutes: array-like, may be negative
    duration_min: DIA in minutes (defaults to 4 h)
    """
    t = np.asarray(t_minutes, dtype=float)
    x = t / duration_min
    frac = np.where(
        (t <= 0) | (t >= duration_min),               # before bolus or after DIA
        0.0,
        1.0 - x**2 * (3 - 2 * x)                      # cubic between 0 and D
    )
    return frac


# --- 2. Minute-level IOB generator ----------------------------------------
def minute_iob(merged, dia_hours=4):
    """
    Returns a DataFrame indexed by minute with an 'IOB' column.
    
    merged['dose']  : Useful columns ['time', 'bolus'] — time is datetime64
    dia_hours : Duration-of-Insulin-Activity in hours (float)
    """
    duration_min = int(dia_hours * 60)

    # 获取包含胰岛素剂量的数据
    dose_data = merged[merged['dose'].notna()].copy()
    if len(dose_data) == 0:
        # 如果没有胰岛素数据，返回全零的IOB
        return pd.Series(0.0, index=merged.index, name='iob')

    # Build a continuous minute timeline covering all boluses + DIA tail
    start = dose_data.index.min().floor('min')
    end   = dose_data.index.max().ceil('min') + pd.Timedelta(minutes=duration_min)
    timeline = pd.DataFrame(index=pd.date_range(start, end, freq='1min'))

    # Collapse boluses that occur within the same minute
    bolus_min =  (dose_data.reset_index()
                 .assign(ts=lambda x: x['ts'].dt.floor('min'))
                 .groupby('ts', as_index=False)['dose'].sum())

    # Prepare arrays for fast broadcasting
    t_index = timeline.index.to_numpy(dtype='datetime64[m]')
    b_times = bolus_min['ts'].to_numpy(dtype='datetime64[m]')
    b_doses = bolus_min['dose'].to_numpy(dtype=float)

    # Matrix of (timeline-minute – bolus-minute) in minutes
    dt = (t_index[:, None] - b_times[None, :]).astype('timedelta64[m]').astype(float)

    # Apply decay kernel and sum each column’s contribution
    frac = iob_fraction(dt, duration_min)
    timeline['iob'] = frac @ b_doses       # matrix mult ⇒ length-N vector

    result = timeline['iob'].reindex(merged.index, method='nearest')
    result.name = 'iob'

    return result


# In[26]:


def cal_delta_iob(merged,start_time, end_time):
    cut = merged.loc[start_time:end_time].copy()
    cut['delta_iob'] = cut['iob'].diff()
    # 修复：只对delta_iob列的负值求和（表示胰岛素吸收量）
    delta_iob = abs(cut[cut['delta_iob']<0]['delta_iob']).sum()

    return delta_iob


# In[27]:


def estimate_meal_size(result_ts, merged):
    # 1. filter 'merged' by 'result_ts'
    retro_detected_cut = merged[merged.index.isin(result_ts)].copy().reset_index()
    # retro_detected_cut = retro_detected_cut[retro_detected_cut['date'] == '2023-11-07']
    cho_est = []

    for i in range(len(retro_detected_cut)):
        a = retro_detected_cut['ts'].iloc[i]
        # iob = retro_detected_cut['iob'].iloc[i]
        bg_before = retro_detected_cut['value'].iloc[i]
        if i == len(retro_detected_cut)-1:
            b = a + pd.Timedelta(hours=4)
            bg_after = merged.loc[merged.index == (a + pd.Timedelta(hours=4)),'value']
            if len(bg_after) > 0:
                bg_after = bg_after.iloc[0]
            
        else:
            b = retro_detected_cut['ts'].iloc[i+1]
            time_diff = b-a
            
            if pd.to_timedelta(time_diff) > pd.Timedelta(hours=4):
                b = a + pd.Timedelta(hours=4)
                bg_after = merged.loc[merged.index == (a + pd.Timedelta(hours=4)),'value']
                if len(bg_after) > 0:
                    bg_after = bg_after.iloc[0]
                
            else: 
                bg_after = retro_detected_cut['value'].iloc[i+1]
                # print(bg_after)

        delta_iob = cal_delta_iob(merged,a, b)
        # print(delta_iob)
        delta_bg = bg_after - bg_before
        cho = delta_iob * icr + delta_bg / gcr
        print(delta_iob,icr,delta_bg,gcr)
        cho_est.append(cho)

    retro_detected_cut['cho_est'] = cho_est

    return retro_detected_cut


# ### UoM Example: patient 2307

# In[ ]:


# # patient 2307 info: needed to estimate meal size
# icr = (12+15)/2         # insulin to carb ratio
# tdd = 17.5              # total daily dose
# isf = 1800/18/tdd      # insulin sensitivity factor
# gcr = isf/icr          # grams of carbs per unit of insulin


# In[28]:


def process_uom_data(patient_id):
    """
    Process UoM data with selectable methods.
    methods: 可选集合/列表，取值可包含：
        {'simple', 'trend', 'lstm_cae', 'decon'}
        为空(None)时默认全部运行
    """
    import os
    out_dir = './results/uom'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing UoM data for patient {patient_id} ...")

    # 读取
    data_dict = read_uom_data(patient_id, ['glucose', 'meal','bolus'])
    glucose_data = data_dict['glucose_data']
    glucose_data = remove_glucose_duplicates(glucose_data)
    meal_data    = data_dict['meal_data']
    meal_data = remove_meal_duplicates(meal_data)
    print(f"meal_data duplicated count: {meal_data['ts'].duplicated().sum()}")
    bolus_data   = data_dict['bolus_data']
    bolus_data   = remove_bolus_duplicates(bolus_data)
    print(f"bolus_data duplicated count: {bolus_data['ts'].duplicated().sum()}")
    # remove the rows where the dose is 0
    bolus_data = bolus_data[bolus_data['dose'] != 0]
    # basal_data   = data_dict['basal_data']
    # print(basal_data.head())

    # 特征与合并
    example    = full_trend_indicators(glucose_data)
    meal_data  = set_ts_as_index(meal_data)
    bolus_data = set_ts_as_index(bolus_data)
    # basal_data = set_ts_as_index(basal_data)
    # basal_data = convert_basal_to_min(basal_data)

    merged = example.join(
        [meal_data[['carbs']], bolus_data[['dose']]],
        how='left'
    )
    merged['iob'] = minute_iob(merged, dia_hours=4)

    # 导出
    # merged.to_csv(f'{out_dir}/overall_data_{patient_id}.csv', index=True)

    # 可视化
    # features = ['value', 'medtronic_20m','dexcom_15m']
    # # 过滤空结果，确保传入的是时间戳序列
    # def to_ts_list(v):
    #     t = pd.to_datetime(v)
    #     return t.tolist() if hasattr(t, "tolist") else [t]

    # rt_clean = {k: to_ts_list(v) for k, v in result_ts.items() if v is not None and len(v) > 0}
    # # rt_clean = {k: pd.to_datetime(v) for k, v in result_ts.items() if v is not None and len(v) > 0}

    # fig = create_interactive_trend_figure(
    #     merged, features,
    #     event_features=['carbs','dose'],
    #     result_timestamps=rt_clean
    # )
    # fig.write_html(f'./plots/uom/interactive_plot_{patient_id}.html')

    return merged



# ### NHS Example (baseline)

# In[29]:


def process_nhs_data(patient_id, variant_label=None, *, dataset_manifest=None):
    """Read a standard CSV bundle; original NHS text import is a separate preparation step."""
    from pipelines.dataset_io import load_manifest, load_record, prepare_record
    path = Path(dataset_manifest) if dataset_manifest else Path(__file__).resolve().parent / "datasets/nhs_v1/manifest.json"
    manifest = load_manifest(path)
    records = [x for x in manifest["records"]
               if str(x.get("participant_id")) == str(patient_id)
               and x.get("variant", x.get("session_id")) == variant_label]
    if len(records) != 1:
        raise ValueError(f"Expected one manifest record for {patient_id}/{variant_label}; found {len(records)}.")
    return prepare_record(load_record(manifest, records[0]))


# In[30]:


def cal_delta_iob(start_time, end_time):
    cut = merged.loc[start_time:end_time].copy()
    cut['delta_iob'] = cut['iob'].diff()
    # 修复：只对delta_iob列的负值求和（表示胰岛素吸收量）
    delta_iob = abs(cut[cut['delta_iob']<0]['delta_iob']).sum()

    return delta_iob


# # Main: Run this!

# # fusion-based method (final one)

# In[70]:


import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Sequence

def _longest_positive_run_duration(t_min: np.ndarray, y: np.ndarray, min_delta: float = 0.0) -> float:
    """
    输入:
      - t_min: 单位为分钟的单调非降时间轴 (len N)
      - y: 数值序列 (len N)
    返回:
      - 窗口内“连续上升”最长时长（分钟），基于 y 的一阶差分 > min_delta
    """
    if len(y) < 2:
        return np.nan
    dy = np.diff(y)
    dt = np.diff(t_min)
    pos = dy > min_delta
    if not np.any(pos):
        return 0.0

    max_run = 0.0
    cur_run = 0.0
    for i in range(len(pos)):
        if pos[i]:
            cur_run += dt[i]
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0.0
    return float(max_run)

def _positive_slope_auc(t_min: np.ndarray, y: np.ndarray, min_delta: float = 0.0) -> float:
    """
    对正的一阶差分做时间加权积分：sum(max(dy,0) * dt)
    表征“上升强度 * 持续时长”。
    """
    if len(y) < 2:
        return np.nan
    dy = np.diff(y)
    dt = np.diff(t_min)
    auc = np.sum(np.maximum(dy - min_delta, 0.0) * dt)
    return float(max(auc, 0.0))

def physio_ra_rise_after_iob(
    window_df: pd.DataFrame,
    model: str,                         # 'bergman' or 'hovorka'
    iob_col: str = 'iob',
    dose_col: str = 'dose',
    ra_pref_cols: str = '_ra_series',  # 按优先级选择可用 RA 列
    min_delta: float = 0.0
) -> Dict[str, float]:
    """
    若窗口内 dose 存在非 NaN 且 >0，则从 IOB 首次 >0 的时刻起，统计对应模型 RA 的
    - 连续上升最长时长（分钟）
    - 正斜率 AUC（sum(max(dRA,0)*dt)）
    否则返回 NaN。
    """
    out = {
        f'{model}_ra_post_iob_rise_len_min': np.nan,
        f'{model}_ra_post_iob_rise_auc': np.nan,
    }

    has_dose = window_df[dose_col].notna().any()  # 是否存在非 NaN
    if not has_dose:
        return out

    # 找 IOB 起点
    iob = window_df[iob_col].fillna(0)
    if not (iob > 0).any():  
        return out
    t0 = iob.index[(iob > 0).to_numpy().argmax()]  # 第一次 >0 的时间戳

    # 选择 RA 列
    ra_col: Optional[str] = None
    col = f'{model}{ra_pref_cols}'
    if col in window_df.columns:
        ra_col = col
 
    if ra_col is None:
        return out

    w = window_df.loc[t0:, [ra_col]].dropna()
    if len(w) < 2:
        return out

    t_min = (w.index - w.index[0]).total_seconds() / 60.0
    y = w[ra_col].astype(float).to_numpy()

    out[f'{model}_ra_post_iob_rise_len_min'] = _longest_positive_run_duration(t_min, y, min_delta=min_delta)
    out[f'{model}_ra_post_iob_rise_auc'] = _positive_slope_auc(t_min, y, min_delta=min_delta)
    return out


# ## Components

# ### Component 1: basic statistics (E_rise-acc)

# In[60]:


import numpy as np
import pandas as pd

def pos_grad_energy(window_df: pd.DataFrame, value_col: str = 'value') -> float:
    """
    E_rise-acc = Σ_u max(0, G'(u)) * 1{ G''(u) >= 0 } * dt(u)
    - 输入 window_df 索引需为 DatetimeIndex，且包含 value_col
    - 返回标量特征：窗口内的“正向加速上升能量”
    """
    if value_col not in window_df.columns or len(window_df) < 3:
        return np.nan

    w = window_df[[value_col]].dropna()
    if len(w) < 3:
        return np.nan

    # 时间轴（分钟）
    t = (w.index - w.index[0]).total_seconds() / 60.0
    g = w[value_col].to_numpy(dtype=float)

    # 一阶/二阶导（用不等间距时间轴计算）
    gdot  = np.gradient(g, t)       # dG/dt (mmol/L per min)
    gddot = np.gradient(gdot, t)    # d²G/dt²

    # 每点时间步长（分钟），用于加权
    dt = np.gradient(t)

    mask = (gddot >= 0.0).astype(float)
    energy = float(np.sum(np.maximum(0.0, gdot) * mask * dt))
    return energy


# ### Component 2: trend indicators

# In[61]:


import numpy as np
import pandas as pd

def positive_trend_auc(window_df: pd.DataFrame, col: str) -> float:
    if col not in window_df.columns or len(window_df) < 2:
        return np.nan
    w = window_df[[col]].dropna()
    if len(w) < 2:
        return np.nan
    # 时间轴（分钟）
    t = (w.index - w.index[0]).total_seconds() / 60.0
    y = w[col].to_numpy(dtype=float)
    y_pos = np.where(y > 0.0, y, 0.0)
    # 梯形法积分：sum((y_i + y_{i+1})/2 * dt)
    dt = np.diff(t)
    auc = np.sum((y_pos[:-1] + y_pos[1:]) * 0.5 * dt)
    return float(auc)

# def trend_auc_medtronic_pos(window_df: pd.DataFrame) -> float:
#     # 兼容不同列名
#     for c in ('medtronic_20m', 'medtronic_trend_20m_num'):
#         if c in window_df.columns:
#             return positive_auc(window_df, c)
#     return np.nan

# def trend_auc_dexcom_pos(window_df: pd.DataFrame) -> float:
#     for c in ('dexcom_15m', 'dexcom_trend_15m_num'):
#         if c in window_df.columns:
#             return positive_auc(window_df, c)
#     return np.nan

def trend_auc_medtronic_pos(window_df: pd.DataFrame) -> float:
    # 兼容不同列名
    if 'medtronic_20m' in window_df.columns:
        return positive_trend_auc(window_df, 'medtronic_20m')
    return np.nan

def trend_auc_dexcom_pos(window_df: pd.DataFrame) -> float:
    if 'dexcom_15m' in window_df.columns:
        return positive_trend_auc(window_df, 'dexcom_15m')
    return np.nan


# ### Component 3: physiology modeling


# In[63]:


import pipelines.physio_kf as pk
from typing import Sequence, Dict, Any, Tuple

def attach_physio_kf_series(
    merged: pd.DataFrame,
    models: Sequence[str] = ('bergman', 'hovorka'),
    filter_type: str = 'ukf',
    impl: str = 'package',
    value_col: str = 'value',
    iob_col: str = 'iob',
    dose_col: str = 'dose',
    insulin_input: str = 'iob_activity',
    reindex_method: str = 'nearest',   # 'nearest' 或 'ffill'
    return_raw: bool = False,
    **kwargs: Any                      # 透传可选阈值等参数，如 ra_thresh_mmol_per_min、g_thresh_mmol ...
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    对每个模型运行 physio_kf_method，并把 <model>_ra_series / <model>_extra_series 加入 merged。
    返回：(merged_with_series, raw_outputs_per_model)
    """
    if not isinstance(merged.index, pd.DatetimeIndex):
        merged = set_ts_as_index(merged)
    merged = merged.sort_index()

    merged_out = merged.copy()
    raw_outputs: Dict[str, Dict[str, Any]] = {}

    for model in models:
        out = pk.physio_kf_method(
            merged,
            model=model,
            filter_type=filter_type,
            impl=impl,
            value_col=value_col,
            iob_col=iob_col,
            dose_col=dose_col,
            insulin_input=insulin_input,
            **kwargs
        )
        raw_outputs[model] = out

        # 构造并对齐时间序列
        idx = pd.to_datetime(out['times'])
        ra = pd.Series(np.asarray(out['ra_series']).squeeze(), index=idx, name=f'{model}_ra_series')
        ex = pd.Series(np.asarray(out['extra_series']).squeeze(), index=idx, name=f'{model}_extra_series')

        if reindex_method == 'nearest':
            ra = ra.reindex(merged_out.index, method='nearest')
            ex = ex.reindex(merged_out.index, method='nearest')
        elif reindex_method in ('ffill', 'pad'):
            ra = ra.reindex(merged_out.index).ffill()
            ex = ex.reindex(merged_out.index).ffill()
        else:
            ra = ra.reindex(merged_out.index)
            ex = ex.reindex(merged_out.index)

        merged_out[ra.name] = ra
        merged_out[ex.name] = ex

    return (merged_out, raw_outputs if return_raw else {})



# In[65]:


def prepare_ra_scores(df: pd.DataFrame,
                      ra_cols: tuple[str, ...] = ("bergman_ra_series", "hovorka_ra_series"),
                      k: float = 1.0,
                      clip_neg: bool = True,
                      smooth_min: int | None = None) -> pd.DataFrame:
    """
    对多个 RA 列分别计算：
      - <name>_pos   : 只保留正值(可选平滑)后的 RA
      - <name>_z     : robust z-score（median + MAD）
      - <name>_score : sigmoid(<name>_z, k)
    其中 <name> 由 ra_cols 派生，例如 'bergman_ra'、'hovorka_ra'。
    """
    out = df.copy()

    for ra_col in ra_cols:
        if ra_col not in out.columns:
            continue

        ra = out[ra_col].astype(float)

        # 1) 只保留“正向吸收”（可选）
        if clip_neg:
            ra = ra.clip(lower=0)

        # 2) 轻度平滑（可选）
        if smooth_min is not None and smooth_min >= 3:
            ra = ra.rolling(window=smooth_min, center=True, min_periods=1).mean()

        name = ra_col.replace("_series", "")  # e.g., 'bergman_ra_series' -> 'bergman_ra'
        out[f"{name}_pos"] = ra

        # 3) robust z + sigmoid
        z = robust_zscore(out[f"{name}_pos"])
        out[f"{name}_z"] = z
        out[f"{name}_score"] = sigmoid(z, k=k)

    return out


# In[66]:


from typing import Sequence, Tuple, Dict, Any
import pandas as pd

def build_ra_features(
    merged: pd.DataFrame,
    models: Sequence[str] = ('bergman', 'hovorka'),
    reindex_method: str = 'nearest',          # 对齐方式：'nearest' 或 'ffill'
    k: float = 1.0,                           # sigmoid 陡峭度
    clip_neg: bool = True,                    # 是否截断负值
    smooth_min: int | None = None,            # 可选平滑窗口(分钟个数)，None=不平滑
    return_raw: bool = False,
    **kf_kwargs: Any                          # 透传给 physio_kf_method 的参数（如阈值）
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    依次执行：
      1) attach_physio_kf_series -> 增加 <model>_ra_series / <model>_extra_series
      2) prepare_ra_scores       -> 为所有 <model>_ra_series 生成 _pos/_z/_score

    返回:
      (df_with_ra_scores, raw_outputs_per_model_if_requested)
    """
    # 1) 跑 KF 并并入序列
    merged_with_series, raw = attach_physio_kf_series(
        merged,
        models=models,
        reindex_method=reindex_method,
        return_raw=True,
        **kf_kwargs
    )

    # 2) 针对所有模型的 *_ra_series 计算分数
    ra_cols = tuple(f"{m}_ra_series" for m in models)
    merged_scored = prepare_ra_scores(
        merged_with_series,
        ra_cols=ra_cols,
        k=k,
        clip_neg=clip_neg,
        smooth_min=smooth_min
    )

    return (merged_scored, raw if return_raw else {})


# In[67]:


# window-level 聚合的方式1
def topk_mean(window_df: pd.DataFrame, col: str, k: int = 12, clip_neg: bool = False) -> float:
    """
    window 内取 top-k 的平均（更稳健），k=12 对应 12min（若1-min采样）。
    若你是每5min采样，则 k=3 对应约15min。
    """
    if col not in window_df.columns:
        return np.nan
    w = window_df[[col]].dropna()
    if len(w) == 0:
        return np.nan

    x = w[col].to_numpy(dtype=float)
    if clip_neg:
        x = np.clip(x, 0.0, None)

    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan

    kk = min(k, len(x))
    # 取最大 kk 个
    topk = np.partition(x, -kk)[-kk:]
    return float(np.mean(topk))



# In[59]:


# --- 2) robust z-score + sigmoid 映射工具
def robust_zscore(s: pd.Series, eps: float = 1e-9) -> pd.Series:
    med = np.nanmedian(s.values)
    mad = np.nanmedian(np.abs(s.values - med))
    scale = 1.4826 * mad  # 将 MAD 近似成标准差
    if not np.isfinite(scale) or scale < eps:
        # MAD 为 0/极小：退化成全 0（无异常度区分）
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - med) / scale

def sigmoid(x: pd.Series, k: float = 1.0) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * x))

def sigmoid_centered(x: pd.Series, k: float = 1.0, z0: float = 2.0) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * (x - z0)))


# ### Pipeline

# In[58]:


import numpy as np
import pandas as pd
from typing import Dict, Callable, Any, Optional, Sequence

def compute_components_only(
    merged: pd.DataFrame,
    window_length_min: int = 90,
    step_min: int = 5,
    component_fns: Optional[Dict[str, Callable[[pd.DataFrame], Any]]] = None,
    include_prob: bool = True,      # 只对 component_fns 里的组件做 z/p
    sigmoid_k: float = 1.0,
    # 新增：生理组件控制项
    ra_models: Sequence[str] = ("bergman", "hovorka"),
    ra_use_score: bool = False,     # False→用 '<model>_ra_pos'，True→用 '<model>_ra_score'
    ra_topk_k: int = 12,             # top-k 个数（1-min 采样下≈12min）
    contiguous_blocks: Optional[Sequence[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    fit_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    返回仅包含各组件的时序：['start','end', <components> ,(可选 z/p)]
    额外：当窗口内 iob 非全零时，计算 2 个生理特征：
      - hovorka_ra_topk, hovorka_ra_auc
    """
    if not isinstance(merged.index, pd.DatetimeIndex):
        merged = set_ts_as_index(merged)
    merged = merged.sort_index()

    if component_fns is None:
        component_fns = {
            'pos_grad_energy': lambda w: pos_grad_energy(w, 'value'),
            'trend_auc_medtronic_pos': trend_auc_medtronic_pos,
            'trend_auc_dexcom_pos':    trend_auc_dexcom_pos
        }

    win  = pd.Timedelta(minutes=window_length_min)
    step = pd.Timedelta(minutes=step_min)

    def _window_is_valid(t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
        if contiguous_blocks is None:
            return True
        for b0, b1 in contiguous_blocks:
            if t0 >= b0 and t1 <= b1:
                w = merged.loc[t0:t1]
                if w.empty:
                    return False
                if len(w) < window_length_min + 1:
                    return False
                diffs = w.index.to_series().diff().dropna()
                if not diffs.empty and (diffs > pd.Timedelta(minutes=1)).any():
                    return False
                return True
        return False

    t_start = merged.index.min().floor('min')
    t_end   = merged.index.max().ceil('min')

    results = []
    t = t_start
    while t + win <= t_end:
        w_end = t + win
        if not _window_is_valid(t, w_end):
            results.append({
                'start': t,
                'end': w_end,
                'pos_grad_energy': np.nan,
                'trend_auc_medtronic_pos': np.nan,
                'trend_auc_dexcom_pos': np.nan,
                'bergman_ra_post_iob_rise_len_min': np.nan,
                'bergman_ra_post_iob_rise_auc': np.nan,
                'hovorka_ra_post_iob_rise_len_min': np.nan,
                'hovorka_ra_post_iob_rise_auc': np.nan,
            })
            t += step
            continue

        w = merged.loc[t: w_end]
        if w.empty:
            t += step
            continue

        feats: Dict[str, Any] = {}
        # 1) 非生理组件
        for name, fn in component_fns.items():
            try:
                feats[name] = fn(w)
            except Exception:
                feats[name] = np.nan

        # 2) 生理组件（条件：窗口内有bolus输入）
        physio_defaults = {
            'bergman_ra_post_iob_rise_len_min': np.nan,
            'bergman_ra_post_iob_rise_auc':     np.nan,
            'hovorka_ra_post_iob_rise_len_min': np.nan,
            'hovorka_ra_post_iob_rise_auc':     np.nan,
        }
        feats.update(physio_defaults)

     
        for model in ra_models:  # ('bergman','hovorka')
            stats = physio_ra_rise_after_iob(
                w,
                model=model,
                iob_col='iob',
                dose_col='dose',
                ra_pref_cols=('_ra_series'),  # 优先用 *_ra_series，若无则 *_ra_pos
                min_delta=0.0
            )
            feats.update(stats)
        

        results.append({'start': t, 'end': t + win, **feats})
        t += step

    df = pd.DataFrame(results)

    # 仅对 component_fns 中的键做 z/p（生理4列若也要 z/p，可把它们加到下面 keys 列表）
    if include_prob:
        for name in component_fns.keys():
            if name in df.columns:
                if fit_mask is not None:
                    ref = pd.to_numeric(df.loc[fit_mask, name], errors="coerce").dropna()
                    med = float(np.nanmedian(ref.values)) if not ref.empty else 0.0
                    mad = float(np.nanmedian(np.abs(ref.values - med))) if not ref.empty else 0.0
                    scale = 1.4826 * mad
                    if not ref.empty and np.isfinite(scale) and scale >= 1e-9:
                        z = (df[name] - med) / scale
                    else:
                        z = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
                else:
                    z = robust_zscore(df[name])
                p = sigmoid(z, k=sigmoid_k)
                df[f'{name}_z'] = z
                df[f'{name}_p'] = p

    return df



