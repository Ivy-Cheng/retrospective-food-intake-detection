import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_nhs_meal_file(file_path):
    """
    解析NHS多餐数据文件，将不同类型的数据提取到不同的DataFrame中
    
    参数:
        file_path: 文件路径（相对或绝对路径）
    
    返回:
        dict: 包含不同数据类型DataFrame的字典
    """
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取整个文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 分割文件内容
    lines = content.split('\n')
    
    # 提取患者信息
    patient_info = {}
    for line in lines[:10]:  # 患者信息通常在前几行
        if 'ID:' in line:
            patient_info['ID'] = line.split('ID:')[1].strip()
        elif 'Weight:' in line:
            weight_str = line.split('Weight:')[1].strip().split()[0]
            patient_info['Weight'] = float(weight_str) if weight_str != '0.000' else np.nan
        elif 'Basal:' in line:
            basal_str = line.split('Basal:')[1].strip().split()[0]
            patient_info['Basal'] = float(basal_str) if basal_str != '0.0' else np.nan
    
    # 解析各个数据段
    def parse_data_section(lines, start_marker, columns):
        """解析数据段"""
        data_list = []
        in_section = False
        
        for line in lines:
            if start_marker in line:
                in_section = True
                continue
            elif line.strip() == '' and in_section:
                # 遇到空行，检查是否是段结束
                continue
            elif '****' in line and in_section:
                # 遇到星号分隔线，段结束
                break
            elif in_section and line.strip() != '' and not line.startswith('Time'):
                # 解析数据行
                parts = line.strip().split('\t')
                if len(parts) >= len(columns):
                    row_data = {}
                    for i, col in enumerate(columns):
                        if i == 0:  # 时间列
                            try:
                                row_data[col] = pd.to_datetime(parts[i], format='%d/%m/%Y %H:%M')
                            except:
                                row_data[col] = parts[i]
                        else:
                            try:
                                # 尝试转换为数值
                                value = parts[i].strip()
                                if value in ['R', 'S', 'N']:
                                    row_data[col] = value
                                else:
                                    row_data[col] = float(value)
                            except:
                                row_data[col] = parts[i].strip()
                    data_list.append(row_data)
        
        return pd.DataFrame(data_list) if data_list else pd.DataFrame()
    
    # 解析各个数据段
    results = {'patient_info': patient_info}
    
    # 餐食数据
    results['enteral_bolus'] = parse_data_section(lines, 'Enteral_bolus', ['Time', 'CHO'])
    results['enteral_infusion'] = parse_data_section(lines, 'Enteral_infusion', ['Time', 'Rate'])
    
    # 胰岛素数据
    results['insulin_bolus'] = parse_data_section(lines, 'Insulin_bolus', ['Time', 'Bolus', 'Duration', 'Insulin'])
    results['insulin_infusion'] = parse_data_section(lines, 'Insulin_infusion', ['Time', 'Rate', 'Insulin'])
    
    # 血糖数据
    results['glucose_concentration'] = parse_data_section(lines, 'Glucose_concentration', ['Time', 'Concentration'])
    
    return results

# def combine_all_data(parsed_data):
#     """
#     将所有数据合并到一个完整的时间序列DataFrame中
    
#     参数:
#         parsed_data: parse_nhs_meal_file返回的字典
    
#     返回:
#         pd.DataFrame: 合并后的完整数据
#     """
#     all_dfs = []
    
#     # 处理餐食数据
#     if not parsed_data['enteral_bolus'].empty:
#         meals = parsed_data['enteral_bolus'].copy()
#         meals.rename(columns={'CHO': 'carbs'}, inplace=True)
#         meals['data_type'] = 'meal'
#         all_dfs.append(meals[['Time', 'carbs', 'data_type']])
    
#     # 处理胰岛素bolus数据
#     if not parsed_data['insulin_bolus'].empty:
#         bolus = parsed_data['insulin_bolus'].copy()
#         bolus.rename(columns={'Bolus': 'insulin_bolus'}, inplace=True)
#         bolus['data_type'] = 'insulin_bolus'
#         all_dfs.append(bolus[['Time', 'insulin_bolus', 'data_type']])
    
#     # 处理胰岛素infusion数据
#     if not parsed_data['insulin_infusion'].empty:
#         infusion = parsed_data['insulin_infusion'].copy()
#         infusion.rename(columns={'Rate': 'insulin_infusion'}, inplace=True)
#         infusion['data_type'] = 'insulin_infusion'
#         all_dfs.append(infusion[['Time', 'insulin_infusion', 'data_type']])
    
#     # 处理血糖数据
#     if not parsed_data['glucose_concentration'].empty:
#         glucose = parsed_data['glucose_concentration'].copy()
#         glucose.rename(columns={'Concentration': 'glucose'}, inplace=True)
#         glucose['data_type'] = 'glucose'
#         all_dfs.append(glucose[['Time', 'glucose', 'data_type']])
    
#     # 合并所有数据
#     if all_dfs:
#         combined = pd.concat(all_dfs, ignore_index=True)
#         combined.sort_values('Time', inplace=True)
#         combined.reset_index(drop=True, inplace=True)
#         return combined
#     else:
#         return pd.DataFrame()

def get_nhs_file_path(filename, base_dir=None):
    """
    获取NHS文件的正确路径
    
    参数:
        filename: 文件名
        base_dir: 基础目录，如果为None则自动检测
    
    返回:
        str: 文件的完整路径
    """
    if base_dir is None:
        # 获取当前脚本的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.getcwd()
        
        # 可能的NHS目录位置
        possible_paths = [
            os.path.join(current_dir, 'NHS', 'multi-meal', filename),  # 如果脚本在根目录
            os.path.join(current_dir, '..', 'NHS', 'multi-meal', filename),  # 如果脚本在子目录
            os.path.join(script_dir, '..', 'NHS', 'multi-meal', filename),  # 基于脚本位置
            os.path.join(os.path.dirname(current_dir), 'NHS', 'multi-meal', filename),  # 其他情况
            os.path.join(os.path.dirname(script_dir), 'NHS', 'multi-meal', filename),  # 基于脚本父目录
        ]
        
        # 检查哪个路径存在
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # 如果都不存在，返回相对路径
        return os.path.join('..', 'NHS', 'multi-meal', filename)
    else:
        return os.path.join(base_dir, 'NHS', 'multi-meal', filename)

# 使用示例
if __name__ == "__main__":
    # 文件名
    filename = "xt_apman02_aspart.txt"
    
    # 方法1：自动检测路径
    file_path = get_nhs_file_path(filename)
    print(f"尝试读取文件: {file_path}")
    
    # 方法2：手动指定基础目录
    # base_directory = r"C:\Users\g69086nc\OneDrive - The University of Manchester\Datasets"
    # file_path = get_nhs_file_path(filename, base_directory)
    
    # 方法3：直接指定完整路径
    # file_path = r"C:\Users\g69086nc\OneDrive - The University of Manchester\Datasets\NHS\multi-meal\xt_apman02_aspart.txt"
    
    try:
        # 解析文件
        print("正在解析NHS数据文件...")
        parsed_data = parse_nhs_meal_file(file_path)
        
        # 显示解析结果
        print(f"\n=== 患者信息 ===")
        for key, value in parsed_data['patient_info'].items():
            print(f"{key}: {value}")
        
        print(f"\n=== 数据段统计 ===")
        for key, df in parsed_data.items():
            if key != 'patient_info' and isinstance(df, pd.DataFrame):
                print(f"{key}: {len(df)} 条记录")
                if not df.empty:
                    print(f"  时间范围: {df['Time'].min()} 到 {df['Time'].max()}")
        
        # 合并所有数据
        print(f"\n=== 合并所有时间序列数据 ===")
        combined_df = combine_all_data(parsed_data)
        print(f"合并后总记录数: {len(combined_df)}")
        
        if not combined_df.empty:
            print(f"时间范围: {combined_df['Time'].min()} 到 {combined_df['Time'].max()}")
            print(f"\n前5条记录:")
            print(combined_df.head())
            
            print(f"\n数据类型分布:")
            print(combined_df['data_type'].value_counts())
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请检查以下可能的解决方案:")
        print("1. 确认文件路径是否正确")
        print("2. 使用绝对路径")
        print("3. 修改脚本中的base_directory变量") 