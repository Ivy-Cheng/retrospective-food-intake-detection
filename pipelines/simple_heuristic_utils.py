# Simple Heuristic Method Pipeline
# Used for retrospective meal detection

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.signal import savgol_filter, find_peaks


def filter_cgm_data(glu_data, day_list, window_length=61, polyorder=6, plot=True):
# def filter_cgm_data(glu_data, day_list, window_length=125, polyorder=3, plot=True):
    """Filter CGM data using Savitzky-Golay filter.
    
    Args:
        glu_data (pd.DataFrame): Glucose data with 'Timestamp' and 'Glucose' columns
        day_list (list): List of dates to filter
        window_length (int): Window length for Savitzky-Golay filter
        polyorder (int): Polynomial order for Savitzky-Golay filter
        plot (bool): Whether to plot the filtered data
        
    Returns:
        pd.DataFrame: Filtered glucose data
    """
    # Filter data for specified days
    cut = glu_data[glu_data['ts'].dt.date.isin(day_list)].reset_index(drop=True)
    
    # Apply Savitzky-Golay filter
    y_filtered = savgol_filter(cut['value'], window_length=window_length, polyorder=polyorder)
    
    # Create output DataFrame
    cut_filtered = cut.copy()
    cut_filtered['ycgm_F'] = y_filtered
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(cut['ts'], cut['value'], label='Original CGM Signal (Noisy)', alpha=0.5)
        plt.plot(cut['ts'], y_filtered, label='Filtered Signal y(t)', color='red')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Glucose Level')
        plt.title('Savitzky-Golay Filter Applied to CGM Data')
        plt.grid(True)
        plt.show()
    
    return cut_filtered

def detect_extrema(filtered_data, plot=True):
    """Detect local minima and maxima in filtered glucose data.
    
    Args:
        filtered_data (pd.DataFrame): Filtered glucose data with 'ycgm_F' column
        plot (bool): Whether to plot the detected extrema
        
    Returns:
        tuple: (maxima_indices, minima_indices)
    """
    ycgm_F = filtered_data['ycgm_F']
    
    # Detect maxima and minima
    maxima_indices, _ = find_peaks(ycgm_F)
    minima_indices, _ = find_peaks(-ycgm_F)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_data['ts'], ycgm_F, label='Glucose Levels', color='blue')
        plt.scatter(filtered_data.iloc[maxima_indices]['ts'], 
                   ycgm_F[maxima_indices], 
                   color='red', 
                   label='Maxima (Peaks)', 
                   marker='^')
        plt.scatter(filtered_data.iloc[minima_indices]['ts'], 
                   ycgm_F[minima_indices], 
                   color='green', 
                   label='Minima (Valleys)', 
                   marker='v')
        plt.xlabel('Time')
        plt.ylabel('Glucose Level')
        plt.legend()
        plt.title('Detected Minima and Maxima in Glucose Levels')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    
    return maxima_indices, minima_indices

def detect_rising_segments(filtered_data, maxima_indices, minima_indices):
    """Detect rising segments in glucose data.
    
    Args:
        filtered_data (pd.DataFrame): Filtered glucose data
        maxima_indices (np.array): Indices of local maxima
        minima_indices (np.array): Indices of local minima
        
    Returns:
        pd.DataFrame: Rising segments information
    """
    rising_segments = []
    
    for i in range(len(minima_indices)):
        tmin_index = minima_indices[i]
        tmax_candidates = maxima_indices[maxima_indices > tmin_index]
        
        if len(tmax_candidates) > 0:
            tmax_index = tmax_candidates[0]
            delta_y = filtered_data['ycgm_F'].iloc[tmax_index] - filtered_data['ycgm_F'].iloc[tmin_index]
            
            # Calculate derivatives
            first_derivative = np.gradient(filtered_data['ycgm_F'].iloc[tmin_index:tmax_index + 1])
            second_derivative = np.gradient(first_derivative)
            
            t1_index = tmin_index + np.argmax(first_derivative)
            t2_index = tmin_index + np.argmax(second_derivative)
            
            rising_segments.append({
                "tmin": filtered_data.iloc[tmin_index]['ts'],
                "tmax": filtered_data.iloc[tmax_index]['ts'],
                "t1": filtered_data.iloc[t1_index]['ts'],
                "t2": filtered_data.iloc[t2_index]['ts'],
                "delta_y": delta_y
            })
    
    return pd.DataFrame(rising_segments)

def detect_meals(filtered_data, rising_segments_df, delta_y_min=20/18, roc_min=0.8/18):
    """Detect meals based on rising segments and thresholds.
    
    Args:
        filtered_data (pd.DataFrame): Filtered glucose data
        rising_segments_df (pd.DataFrame): Rising segments information
        delta_y_min (float): Minimum glucose hub threshold
        roc_min (float): Minimum rate of change threshold
        
    Returns:
        pd.DataFrame: Detected meal times and related information
    """
    meal_detections = []
    first_derivative = np.gradient(filtered_data["ycgm_F"].to_numpy())
    
    for _, segment in rising_segments_df.iterrows():
        tmin = segment["tmin"]
        t2 = segment["t2"]
        delta_y = segment["delta_y"]
        tmax = segment['tmax']
        
        # Compute rate of change at t1
        t1_index = filtered_data[filtered_data["ts"] == segment["t1"]].index[0]
        glucose_values = filtered_data["ycgm_F"].to_numpy()
        first_derivative = np.gradient(glucose_values)
        roc_t1 = first_derivative[t1_index]
        
        # Calculate area under curve
        glu_seg = filtered_data[(filtered_data['ts']>=tmin) & 
                               (filtered_data['ts']<=tmax)]
        baseline_glucose = glu_seg["ycgm_F"].iloc[0]
        adjusted_glucose = glu_seg["ycgm_F"] - baseline_glucose
        time_numeric = (glu_seg['ts'] - glu_seg['ts'].iloc[0]).dt.total_seconds() / 60
        auc = scipy.integrate.trapezoid(adjusted_glucose, time_numeric)
        
        # Check meal detection conditions
        if delta_y > delta_y_min and roc_t1 > roc_min:
            tmeal = pd.to_datetime((tmin.timestamp() + t2.timestamp()) / 2, unit='s')
            meal_detections.append({
                "tmeal": tmeal,
                "tmin": tmin,
                "t2": t2,
                "delta_y": delta_y,
                "roc_t1": roc_t1,
                "auc": auc
            })
    
    return pd.DataFrame(meal_detections) 



def simple_heuristic(glu_data, day_list, *, mask_gaps=True, plot=False):
    """Run the original baseline independently on unmasked prepared-CGM segments.

    Preserve the 125-sample cubic Savitzky-Golay filter and original detection
    thresholds. Segments shorter than its fixed filter window are not processed.
    """
    from pipelines.gap_masking import prepared_gap_intervals, split_processing_blocks
    data = glu_data.copy()
    data["ts"] = pd.to_datetime(data["ts"])
    data = data.sort_values("ts").reset_index(drop=True)
    columns = ["tmeal", "tmin", "t2", "delta_y", "roc_t1", "auc"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    # Derive masks before day selection, so omitted days cannot be concatenated.
    masks = prepared_gap_intervals(data) if mask_gaps else []
    selected = data[data.ts.dt.date.isin(day_list)].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    if mask_gaps:
        masks += prepared_gap_intervals(selected)
    blocks = split_processing_blocks([(selected.ts.min(), selected.ts.max())], masks)
    results = []
    for start, end in blocks:
        part = selected[(selected.ts >= start) & (selected.ts <= end)].copy()
        if len(part) < 125:
            continue
        filtered = filter_cgm_data(part, part.ts.dt.date.unique(), window_length=125, polyorder=3, plot=plot)
        maxima, minima = detect_extrema(filtered, plot=plot)
        rising = detect_rising_segments(filtered, maxima, minima)
        if rising.empty:
            continue
        events = detect_meals(filtered, rising, delta_y_min=20/18, roc_min=0.8/18)
        if not events.empty:
            results.append(events)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=columns)
