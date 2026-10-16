"""Original glucose preprocessing helpers extracted from the sibling dependency.

Original numerical interpolation is retained. Standard CSV readers use this
module directly. Attribution/ownership must be confirmed before release.
"""
import pandas as pd


def missing_chunks(glucose_data):
    glucose_data["ts"] = pd.to_datetime(glucose_data["ts"],format="%d-%m-%Y %H:%M:%S")
    glucose_data["ts_diff_minutes"] = glucose_data["ts"].diff().dt.total_seconds() / 60
    glucose_data["date"] = glucose_data["ts"].dt.date

    # --- Flag missing chunks -----------------------------------------------------------------------#
    glucose_data["missing_flag"] = None
    # The current row (i+1): where ts_diff_minutes > 30 (indicates a gap)
    gap_condition = (glucose_data["ts_diff_minutes"] > 30) & (glucose_data["ts_diff_minutes"] <= 216)

    # Row i -> 'missing_start' (the row *before* the gap, i.e., shift back by 1)
    glucose_data.loc[gap_condition.shift(-1).fillna(False), "missing_flag"] = "starts"

    # mask_start = gap_condition.shift(-1).fillna(False).astype(bool)
    # glucose_data.loc[mask_start, "missing_flag"] = "starts"

    # Row i+1 -> 'missing_end' (the row *after* the gap)
    glucose_data.loc[gap_condition, "missing_flag"] = "ends"

    # Identify and drop days with large gaps (>216 mins) ---
    dates_to_drop = glucose_data.loc[
        glucose_data["ts_diff_minutes"] > 216,
        "date"
    ].unique()
    print("Days dropped due to large gaps (>216 minutes):", dates_to_drop)

    # Drop those days
    glucose_data = glucose_data[~glucose_data["date"].isin(dates_to_drop)].reset_index(drop=True)

    return glucose_data


def interpolation(df):
    df["ts"] = df["ts"].dt.floor('min')
    df = df.sort_values("ts").reset_index(drop=True)
   
    start_time = df["ts"].iloc[0]
    end_time = df["ts"].iloc[-1]
    timestamps = pd.date_range(start=start_time, end=end_time, freq="1min")

    new_df = pd.DataFrame({"ts": timestamps})
    new_df["ts"] = pd.to_datetime(new_df["ts"])
    new_df = new_df.merge(df, on="ts", how="left")
    new_df["value"] = pd.to_numeric(new_df["value"], errors="coerce")
    new_df["value"] = new_df["value"].interpolate(method="linear")

    drop_indices = []
    inside_block = False

    for idx, val in new_df["missing_flag"].items():
        if val == "starts":
            inside_block = True
        elif val == "ends":
            inside_block = False
        elif inside_block:
            drop_indices.append(idx)

    df_cleaned = new_df.drop(index=drop_indices).reset_index(drop=True)
    df_cleaned = df_cleaned.drop(columns=["ts_diff_minutes"])
    df_cleaned["date"] = pd.to_datetime(df_cleaned["ts"]).dt.date
    df_cleaned["missing_flag"] = df_cleaned["missing_flag"].fillna("None")
    return df_cleaned


def process_glucose_by_date(glucose_data, *, strict=True):
    """
    Wrapper function that processes glucose data day by day using the original functions
    """
    # 首先解析时间戳并添加日期列
    glucose_data = glucose_data.copy()
    glucose_data["ts"] = pd.to_datetime(glucose_data["ts"], format="%d-%m-%Y %H:%M:%S")
    glucose_data["date"] = glucose_data["ts"].dt.date
    
    processed_days = []
    
    # 按日期分组处理
    for date, day_data in glucose_data.groupby("date"):
        print(f"Processing day: {date}")
        
        # 重置索引，准备单独处理这一天的数据
        day_data = day_data.reset_index(drop=True)
        
        # 为了使用原有函数，需要临时移除已有的date列
        day_data_for_processing = day_data.drop(columns=["date"]).copy()
        
        # 应用原有的 missing_chunks 函数
        try:
            day_after_missing_chunks = missing_chunks(day_data_for_processing)
            
            # 检查这一天是否被丢弃（空DataFrame）
            if len(day_after_missing_chunks) == 0:
                print(f"Day {date} was dropped due to large gaps")
                continue
                
            # 应用原有的 interpolation 函数
            day_after_interpolation = interpolation(day_after_missing_chunks)
            
            # 添加处理好的这一天到列表中
            if len(day_after_interpolation) > 0:
                processed_days.append(day_after_interpolation)
                
        except Exception as e:
            if strict:
                raise ValueError(f"Preprocessing failed for {date}: {e}") from e
            print(f"Error processing day {date}: {e}")
            continue
    
    # 合并所有处理好的天数
    if processed_days:
        final_df = pd.concat(processed_days, ignore_index=True)
        # 确保按时间排序
        final_df = final_df.sort_values("ts").reset_index(drop=True)
        return final_df
    else:
        return pd.DataFrame()
