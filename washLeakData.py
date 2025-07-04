import os
from pathlib import Path
import shutil
import pandas as pd
import soundfile as sf
import pdb
import numpy as np
# 定义要搜索的根目录

needDataAlignment = False # 是否需要对数据进行时间对齐, 如果对齐, 那么就是对每一个实验生成一个csv文件, 否则就是对每个实验生成两个csv文件, # 一个是声音数据, 一个是其他数据

root_directory_str = "/home/zhangzhe/data/leak" # os.walk 通常接受字符串路径
class RAWSource(object):
    def __init__(self, **kwargs):
        super(RAWSource, self).__init__(**kwargs)

    def raw_to_time_series(self, raw_file_path, channels=1, samplerate=8000,
                           subtype='PCM_32', endian='LITTLE'):
        """
        Reads a RAW acoustic file and returns its time series as two arrays:
        relative time (in seconds) and signal values.

        Args:
            raw_file_path (str or Path): Path to the RAW file.
            channels (int): Number of audio channels.
            samplerate (int): Sample rate of the audio (samples per second).
            subtype (str): Subtype of the audio data (e.g., 'PCM_32', 'FLOAT').
            endian (str): Endianness of the data ('LITTLE', 'BIG', 'FILE').

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Time array (in seconds, relative to start).
                - numpy.ndarray: Signal values array.
        """
        try:
            # sf.read returns (data, samplerate_from_file_header_if_any)
            # When reading RAW, samplerate is given as an argument, so the second element
            # is usually the specified samplerate.
            signal_raw, actual_samplerate = sf.read(raw_file_path, channels=channels,
                                                    samplerate=samplerate, subtype=subtype,
                                                    endian=endian)

            signal = signal_raw # This is already the value array
            
            # Generate the time array
            time_array = np.arange(len(signal)) / actual_samplerate

            print(f"Successfully read {raw_file_path}. Sample rate: {actual_samplerate}, Samples: {len(signal)}")
            return time_array, signal

        except Exception as e:
            print(f"Error reading {raw_file_path}: {e}")
            return None, None

# --- 函数：读取单个 RAW 文件并转换为 Pandas DataFrame ---
def read_single_raw_file_to_df(raw_file_path, **kwargs):
    """
    Reads a single RAW acoustic file, processes it, and returns a pandas DataFrame.

    Args:
        raw_file_path (str or Path): Path to the RAW file.
        **kwargs: Additional arguments to pass to RAWSource.raw_to_time_series
                  (e.g., channels, samplerate, subtype, endian).

    Returns:
        pandas.DataFrame: A DataFrame with 'time_in_seconds' as index and 'value' column,
                          or None if an error occurred.
    """
    raw_reader = RAWSource()
    time_array, signal_values = raw_reader.raw_to_time_series(raw_file_path, **kwargs)

    if time_array is not None and signal_values is not None:
        # 创建一个 Pandas Series 或 DataFrame
        #索引改为条目行数
        df_signal = pd.DataFrame(signal_values, columns=['value'])
        df_signal['time_in_seconds'] = time_array
        return df_signal
    else:
        return None


def interpolate_dataframe_by_times(df_main: pd.DataFrame, 
                                   time_column: str, 
                                   df_target_times: pd.DataFrame, 
                                   target_time_column: str,
                                   interpolation_method: str = 'linear',
                                   extrapolate: bool = False,
                                   extrapolation_method: str = 'nearest') -> pd.DataFrame:
    """
    根据给定的目标时间序列，对主DataFrame中的数据进行线性插值。

    Args:
        df_main (pd.DataFrame): 包含原始数据和时间的DataFrame。
                                  时间列必须是可排序的数值类型（如浮点数）。
        time_column (str): df_main中作为时间基准的列名。
        df_target_times (pd.DataFrame): 包含新时间序列的DataFrame。
        target_time_column (str): df_target_times中作为新时间基准的列名。
        interpolation_method (str): 插值方法，默认为'linear'。
                                    其他选项如 'nearest', 'polynomial', 'spline'等。
                                    详情见 pandas.DataFrame.interpolate 文档。
        extrapolate (bool): 是否对外推（即超出原始时间范围）的点进行处理。
                            如果为True，将使用 extrapolation_method 进行外推。
                            默认为False，超出范围的点将是NaN。
        extrapolation_method (str): 外推方法，仅当extrapolate为True时有效。
                                    默认为'nearest'。常用'ffill'（向前填充），'bfill'（向后填充）。

    Returns:
        pd.DataFrame: 一个新的DataFrame，其索引是目标时间序列，列是插值后的数据。
                      如果目标时间点超出原始时间范围且extrapolate为False，
                      则这些点的数据可能为NaN。
    """
    if time_column not in df_main.columns:
        raise ValueError(f"'{time_column}' not found in df_main columns.")
    if target_time_column not in df_target_times.columns:
        raise ValueError(f"'{target_time_column}' not found in df_target_times columns.")

    # 1. 准备主 DataFrame：将时间列设为索引，并排序
    # 创建一个副本以避免修改原始df_main
    df_main_indexed = df_main.set_index(time_column).sort_index()

    # 移除时间列本身，只保留数据列
    data_columns = [col for col in df_main_indexed.columns if col != time_column]
    df_main_data = df_main_indexed[data_columns]

    # 2. 提取目标时间序列
    new_time_points = df_target_times[target_time_column].unique()
    new_time_points.sort() # 确保目标时间点也是排序的

    # 3. 合并原始索引和新时间点，创建新的完整索引
    combined_index = pd.Index(np.unique(np.concatenate([df_main_data.index, new_time_points])),
                              name=time_column) # 保留索引名称

    # 4. 重新索引 df_main_data 到这个新的完整索引
    df_reindexed = df_main_data.reindex(combined_index)

    # 5. 进行插值
    df_interpolated_full = df_reindexed.interpolate(method=interpolation_method)

    # 6. 处理外推
    if not extrapolate:
        # 移除原始数据时间范围之外的插值结果
        # 找到原始数据的最小和最大时间
        original_min_time = df_main_data.index.min()
        original_max_time = df_main_data.index.max()
        
        # 过滤掉在新时间点中超出原始范围的点
        # 这里实际上 interpolate 默认就不会外推，这一步是为了明确行为
        # 并且确保最终结果只包含在原始数据覆盖范围内插值得到的点
        df_interpolated_full = df_interpolated_full.loc[
            (df_interpolated_full.index >= original_min_time) & 
            (df_interpolated_full.index <= original_max_time)
        ]
    else:
        # 如果需要外推，应用外推方法
        # 注意：ffill和bfill会填充NaN，包括外推的NaN
        if extrapolation_method == 'ffill':
            df_interpolated_full = df_interpolated_full.fillna(method='ffill')
        elif extrapolation_method == 'bfill':
            df_interpolated_full = df_interpolated_full.fillna(method='bfill')
        elif extrapolation_method == 'nearest':
             # nearest method for interpolate covers both interpolation and simple extrapolation
             # but to explicitly handle boundary NaN, ffill/bfill might still be needed
             # A more robust nearest extrapolation would be:
            df_interpolated_full = df_interpolated_full.interpolate(method='nearest', fill_value='extrapolate', limit_direction='both')
        else:
            print(f"Warning: Extrapolation method '{extrapolation_method}' not explicitly handled for boundary NaNs. Some NaNs might remain if not covered by interpolation_method.")
            # 对于其他更复杂的插值方法，fill_value='extrapolate' 或手动填充
            # df_interpolated_full = df_interpolated_full.interpolate(method=interpolation_method, fill_value='extrapolate')
            # 对于更复杂的插值，如果 interpolation_method 不支持外推，可能需要额外的ffill/bfill
            # df_interpolated_full = df_interpolated_full.ffill().bfill() # 最简单的两端填充

    # 7. 从插值后的 DataFrame 中提取目标时间点的数据
    # 使用 .loc 再次通过 new_time_points 的值来选择行

    interpolated_df_result = df_interpolated_full.loc[new_time_points]

    # 重置索引，如果希望时间列回到普通列
    # interpolated_df_result = interpolated_df_result.reset_index()

    return interpolated_df_result

print(f"\n--- Using os.walk() combined with pathlib.Path to find all CSV files in '{root_directory_str}' ---")

found_csv_files_os_walk = []

if os.path.exists(root_directory_str) and os.path.isdir(root_directory_str):
    for dirpath, dirnames, filenames in os.walk(root_directory_str):
        # dirpath 是当前遍历到的目录的路径
        # filenames 是当前目录下的所有文件名 (不包含路径)

        for filename in filenames:
            if filename.endswith('.csv'):
                # 构建完整的Path对象
                
                full_path = Path(dirpath) / filename
                found_csv_files_os_walk.append(full_path)
    else:
        print("No CSV files found.")
else:
    print(f"Directory '{root_directory_str}' does not exist or is not a directory. Please run the setup script first.")

dfs = {}



for csv_file in found_csv_files_os_walk:
    csv_file_name = csv_file.name
    csv_file_name_ele = csv_file_name.split('_')
    condition = f"{csv_file_name_ele[0]}_{csv_file_name_ele[1]}_{csv_file_name_ele[2]}"
    df = pd.read_csv(csv_file, low_memory=True)
    df.rename(columns={'Sample': 'Time' + csv_file_name_ele[-1].split(".")[0]}, inplace=True)
    df.rename(columns={'Value': 'Value' + csv_file_name_ele[-1].split(".")[0]}, inplace=True)
    df = df[1:]
    if condition in dfs:
        dfs[condition].append(df)
    else:
        dfs[condition] = [df]



for condition, df_list in dfs.items():
    minValue = min(df.shape[0] for df in df_list)
    df_list = [df.head(minValue) for df in df_list]
    dfs[condition] = pd.concat(df_list, axis=1)
    

for condition, df in dfs.items(): 
    dfs[condition] = df[df['TimeA1'] <= 30]

for condition, df in dfs.items():
    dfs[condition] = df.drop(columns=["TimeA2" , "TimeP1", "TimeP2"])

print(f"\n--- Using os.walk() combined with pathlib.Path to find all RAW files in '{root_directory_str}' ---")

rawFiles = []
if os.path.exists(root_directory_str) and os.path.isdir(root_directory_str):
    for dirpath, dirnames, filenames in os.walk(root_directory_str):
        # dirpath 是当前遍历到的目录的路径
        # filenames 是当前目录下的所有文件名 (不包含路径)

        for filename in filenames:
            if filename.endswith('.raw'):
                # 构建完整的Path对象
                
                full_path = Path(dirpath) / filename
                rawFiles.append(full_path)
    else:
        print("No RAW files found.")
else:
    print(f"Directory '{root_directory_str}' does not exist or is not a directory. Please run the setup script first.")

Hdf = {}

print(f"\n--- Processing RAW files ---")
for raw_file in rawFiles:
    raw_file_name = raw_file.name
    raw_file_name_ele = raw_file_name.split('_')
    if raw_file.name.startswith("Background"):
        continue
    if raw_file_name_ele[-2] == "NN":
        continue
    condition = f"{raw_file_name_ele[0]}_{raw_file_name_ele[1]}_{raw_file_name_ele[2]}" 
    timeColName = 'Time' + raw_file_name_ele[-1].split(".")[0] + raw_file_name_ele[-2]
    df = read_single_raw_file_to_df(raw_file)
    df.rename(columns={'value': 'Value' + raw_file_name_ele[-1].split(".")[0] + raw_file_name_ele[-2]}, inplace=True)
    df.rename(columns={'time_in_seconds': timeColName}, inplace=True)
    if needDataAlignment:
        interpolated_sound_df = interpolate_dataframe_by_times(
            df_main=df,
            time_column='time_in_seconds',
            df_target_times=dfs[condition],
            target_time_column='TimeA1',
        
            )
        interpolated_sound_df = interpolated_sound_df.reset_index(drop=True) 
        interpolated_sound_df.index = interpolated_sound_df.index + 1
        dfs[condition] =  pd.concat([interpolated_sound_df,dfs[condition]], axis=1)
    else:
        if condition not in Hdf:
            Hdf[condition] = df[df[timeColName] <= 30]
        else:
            Hdf[condition] = pd.concat([Hdf[condition], df[df[timeColName] <= 30]], axis=1)


for condition, df in Hdf.items():
    df = df.drop(columns = ["TimeH2N"])        
    Hdf[condition] = df 

pdb.set_trace()

print(f"\n--- Processing completed. Numberlize  ---")
newDFS = {}
for condition, df in dfs.items():
    if condition.endswith("Transient"):
        continue
    newDFS[condition] = dfs[condition]
    condition_ele = condition.split('_')
    if condition_ele[0] == "LO":
        newDFS[condition]['Structure'] = 0
    elif condition_ele[0] == "BR":
        newDFS[condition]['Structure'] = 1
    if condition_ele[1] == "CC":
        newDFS[condition]['LeakType'] = 0
    elif condition_ele[1] == "GL":
        newDFS[condition]['LeakType'] = 1
    elif condition_ele[1] == "LC":
        newDFS[condition]['LeakType'] = 2
    elif condition_ele[1] == "OL":
        newDFS[condition]['LeakType'] = 3
    elif condition_ele[1] == "NL":
        newDFS[condition]['LeakType'] = 4
    if condition_ele[2] == "0.47 LPS":
        newDFS[condition]['FlowCondition'] = 0
    elif condition_ele[2] == "0.18 LPS":
        newDFS[condition]['FlowCondition'] = 1
    elif condition_ele[2] == "ND":
        newDFS[condition]['FlowCondition'] = 2
    
save_DIR = "/home/zhangzhe/data/processed_leak_data/"
count = 1
if needDataAlignment:
    for _, df in newDFS.items():
        df.to_csv(save_DIR + f"leak_exp_{count}.csv", index=False)
        count += 1
else:
    for condition, df in newDFS.items():
        df.to_csv(save_DIR + f"leak_exp_Nsound_{count}.csv", index=False)
        Hdf[condition].to_csv(save_DIR + f"leak_exp_sound_{count}.csv", index=False)

        
        count += 1