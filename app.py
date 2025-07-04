import glob
import os
import time
import math, random

import streamlit as st
import pandas as pd
import numpy as np
import torch

# --- 从项目中导入必要的模块 ---
# 确保 config.py 在同一目录或可通过 PYTHONPATH 访问
from config import config, model_config
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference

# --- 应用配置 ---
st.set_page_config(page_title="管道状态预测模拟", layout="centered")

MODEL_PATH = "FFTbest_classification_model.pth"
CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]
NUM_PIPELINES = 3  # 我们将模拟的管道数量

# --- 核心模拟参数 ---
# 每个数据块的大小 (0.01s 的数据)
CHUNK_SIZE = config.get("window_size", 256) 
# 每个数据块到达的间隔时间（秒）
CHUNK_UPDATE_INTERVAL_SECONDS = 0.2 # 可以加快模拟速度

# --- 关键函数 ---

@st.cache_resource
def load_model_and_data():
    """
    加载预训练模型和样本数据。此函数被缓存以提高性能。
    """
    try:
        predictor = ModelInference(
            model_path=MODEL_PATH,
            config=config,
            model_config=model_config,
            class_names=CLASS_NAMES
        )
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.error(f"请确保 '{MODEL_PATH}' 在项目根目录且 'config.py' 配置正确。")
        return None, None, None

    try:
        data_dir = config["data_dir"]
        file_pattern = config["file_pattern"]
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if not file_paths:
            st.error(f"在 '{data_dir}' 中未找到匹配 '{file_pattern}' 的文件。")
            return predictor, None

        all_dataframes = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return predictor, None

    try:
        if len(all_dataframes) < NUM_PIPELINES:
            st.warning(f"数据文件不足 {NUM_PIPELINES} 个，将使用所有可用文件。")
            selected_indices = list(range(len(all_dataframes)))
        else:
            selected_indices = random.sample(range(len(all_dataframes)), NUM_PIPELINES)

        pipeline_data = [all_dataframes[idx] for idx in selected_indices]

    except Exception as e:
        st.error(f"样本选择失败: {e}")
        return predictor, None

    return predictor, pipeline_data

# --- Streamlit 应用主逻辑 ---

# 1. 加载模型和数据 (只运行一次)
predictor, pipeline_data = load_model_and_data()
if not predictor or not pipeline_data:
    st.stop()

# 2. 初始化会话状态 (Session State)
if 'selected_pipeline_index' not in st.session_state:
    st.session_state.selected_pipeline_index = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# 3. 侧边栏：用于选择管道
with st.sidebar:
    st.header("管道控制面板")
    pipeline_names = [f"Pipeline {i+1}" for i in range(len(pipeline_data))]

    # 当用户点击不同的管道时，更新会话状态并重置模拟
    for i, name in enumerate(pipeline_names):
        if st.button(name, key=f"pipe_{i}"):
            if st.session_state.selected_pipeline_index != i:
                st.session_state.selected_pipeline_index = i
                # 重置准确率计数器
                st.session_state.correct_predictions = 0
                st.session_state.total_predictions = 0
                st.rerun()

# --- 4. 主界面布局和模拟 ---

# 获取当前选择的管道数据
selected_idx = st.session_state.selected_pipeline_index
selected_df = pipeline_data[selected_idx]

# 提取真实标签和特征数据
target_col = config["target_col"]
feature_cols_names = [col for col in selected_df.columns if col != target_col and col not in config.get("delete_col", [])]

true_label_idx = selected_df[target_col].iloc[0]
true_label = CLASS_NAMES[true_label_idx]
full_df = selected_df[feature_cols_names]
max_timestep = len(full_df)
total_chunks = math.ceil(max_timestep / CHUNK_SIZE)


st.title(f"管道 {selected_idx + 1} 状态预测模拟")
st.markdown("---")

# --- 5. 创建用于动态更新的占位符 ---
status_placeholder = st.empty()
accuracy_placeholder = st.empty()
result_placeholder = st.empty()

# --- 6. 实时模拟循环 ---
# 按 CHUNK_SIZE 步长进行迭代
for i in range(total_chunks):
    start_idx = i * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE
    
    # 获取当前数据块
    chunk_df = full_df.iloc[start_idx:end_idx]

    # 如果数据块小于窗口大小，则跳过（通常是最后一个块）
    if len(chunk_df) < CHUNK_SIZE:
        continue
    
    # 使用当前数据块进行预测
    predicted_label = predictor.predict(chunk_df)

    # 更新准确率计数器
    st.session_state.total_predictions += 1
    if predicted_label == true_label:
        st.session_state.correct_predictions += 1
    
    # --- 更新界面元素 ---
    status_text = "🟢 **模拟进行中...**" if i < total_chunks - 1 else "🏁 **模拟完成!**"
    status_placeholder.header(f"{status_text}")
    status_placeholder.progress((i + 1) / total_chunks, text=f"已处理数据块: {i + 1} / {total_chunks}")

    # 更新准确率显示
    correct = st.session_state.correct_predictions
    total = st.session_state.total_predictions
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    with accuracy_placeholder.container():
        st.metric(label="**实时预测准确率**", value=f"{accuracy:.2f} %", delta=f"{correct} / {total} 正确")

    # 更新最新的预测结果
    with result_placeholder.container():
        st.subheader("最新预测结果")
        col_pred, col_true = st.columns(2)
        with col_pred:
            st.write("**模型预测:**")
            if predicted_label == true_label:
                st.success(f"✔️ {predicted_label}")
            else:
                st.error(f"❌ {predicted_label}")
        with col_true:
            st.write("**真实状态:**")
            st.info(f"ℹ️ {true_label}")
        
        st.write("**用于本次预测的数据块 (最后5行):**")
        st.dataframe(chunk_df.tail(5))


    # 控制模拟速度
    time.sleep(CHUNK_UPDATE_INTERVAL_SECONDS)

# 循环结束后，确保最终状态可见
total = st.session_state.total_predictions
if total == 0:
    st.warning("没有足够的数据来生成任何预测。")
else:
    status_placeholder.header("🏁 **模拟完成!**")