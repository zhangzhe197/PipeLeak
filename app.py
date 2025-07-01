import streamlit as st
import torch
import time
import os
import glob
import random

# --- 从项目中导入必要的模块 ---
from config import config, model_config
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference

# --- 应用配置 ---
st.set_page_config(page_title="多管道切换预测模拟 (v4)", layout="centered")

MODEL_PATH = "FFTbest_classification_model.pth"
CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]
NUM_PIPELINES = 100  # 要随机选择的管道（文件）数量
SIMULATION_INTERVAL_SECONDS = 0 # 模拟速度

# --- 关键函数 (缓存以提高性能) ---

@st.cache_resource
def load_resources():
    """
    加载模型和数据。
    1. 随机选择 N 个数据文件。
    2. 为每个文件创建一个独立的 TimeSeriesDataset 对象。
    3. 存储每个管道的 Dataset 对象、样本数和真实标签。
    """
    # 1. 初始化模型推理接口
    try:
        predictor = ModelInference(
            model_path=MODEL_PATH,
            config=config,
            model_config=model_config,
            class_names=CLASS_NAMES
        )
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None, None

    # 2. 查找并随机选择文件
    try:
        all_file_paths = sorted(glob.glob(os.path.join(config["data_dir"], config["file_pattern"])))
        if not all_file_paths:
            st.error(f"在 '{config['data_dir']}' 中未找到匹配 '{config['file_pattern']}' 的文件。")
            return predictor, None

        if len(all_file_paths) < NUM_PIPELINES:
            st.warning(f"文件不足 {NUM_PIPELINES} 个，将使用所有 {len(all_file_paths)} 个可用文件。")
            paths_to_simulate = all_file_paths
        else:
            paths_to_simulate = random.sample(all_file_paths, NUM_PIPELINES)
        
        pipeline_info = []
        st.info("正在为每个选中的管道创建独立的 Dataset...")

        for path in paths_to_simulate:
            # --- 核心技巧：使用文件名作为 file_pattern 来加载单个文件 ---
            data_dir = os.path.dirname(path)
            file_pattern = os.path.basename(path)
            
            dataset_for_pipeline = TimeSeriesDataset(
                data_dir=data_dir,
                file_pattern=file_pattern, # <--- 正确用法
                window_size=config["window_size"],
                stride=config["stride"],
                target_col=config["target_col"],
                Normalization=config.get("Normalization", True),
                delete_col=config["delete_col"],
                constant_col=config["constant_col"]
            )
            
            # 如果该文件能生成有效样本，则存储其信息
            if len(dataset_for_pipeline) > 0:
                _, label_tensor = dataset_for_pipeline[0] # 从第一个样本获取标签
                pipeline_info.append({
                    'path': file_pattern,
                    'dataset': dataset_for_pipeline, # <--- 直接存储Dataset对象
                    'num_samples': len(dataset_for_pipeline),
                    'true_label_idx': label_tensor.item()
                })
        
        if not pipeline_info:
            st.error("所有选中的文件都无法生成有效的数据样本。")
            return predictor, None

        st.success("模型和数据资源加载完成！")
        return predictor, pipeline_info

    except Exception as e:
        st.error(f"处理数据时发生错误: {e}")
        return predictor, None

# --- Streamlit 应用主逻辑 ---

# 1. 加载所有资源
predictor, pipeline_info = load_resources()
if not predictor or not pipeline_info:
    st.stop()

# 2. 初始化会话状态
if 'selected_pipeline_index' not in st.session_state:
    st.session_state.selected_pipeline_index = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# 3. 侧边栏：用于选择管道
with st.sidebar:
    st.header("管道控制面板")
    pipeline_names = [f"管道 {i+1} ({info['path']})" for i, info in enumerate(pipeline_info)]

    for i, name in enumerate(pipeline_names):
        if st.button(name, key=f"pipe_{i}"):
            if st.session_state.selected_pipeline_index != i:
                st.session_state.selected_pipeline_index = i
                st.session_state.correct_predictions = 0
                st.session_state.total_predictions = 0
                st.rerun()

# --- 4. 主界面布局和模拟 ---

# 获取当前选择的管道信息
selected_idx = st.session_state.selected_pipeline_index
current_pipeline = pipeline_info[selected_idx]

# 从管道信息中直接获取 Dataset 对象和相关参数
dataset_to_simulate = current_pipeline['dataset']
true_label = CLASS_NAMES[current_pipeline['true_label_idx']]
total_samples_in_pipeline = current_pipeline['num_samples']

st.title(f"正在模拟: {current_pipeline['path']}")
st.markdown(f"**真实状态:** `{true_label}` | **此管道样本总数:** `{total_samples_in_pipeline}`")
st.markdown("---")

# 5. 创建用于动态更新的占位符
status_placeholder = st.empty()
accuracy_placeholder = st.empty()
result_placeholder = st.empty()

# 6. 实时模拟循环，直接遍历当前管道的 Dataset
for i, (inputs_tensor, _) in enumerate(dataset_to_simulate):
    # --- 模型预测 ---
    inputs_tensor = inputs_tensor.unsqueeze(0).to(predictor.device)
    
    with torch.no_grad():
        outputs = predictor.model(inputs_tensor)
    
    _, predicted_idx = torch.max(outputs.data, 1)
    predicted_label = predictor.class_names[predicted_idx.item()]

    # 更新准确率计数器
    st.session_state.total_predictions += 1
    if predicted_label == true_label:
        st.session_state.correct_predictions += 1
    
    # --- 更新界面元素 ---
    status_text = "🟢 **模拟进行中...**" if i < total_samples_in_pipeline - 1 else "🏁 **模拟完成!**"
    status_placeholder.header(f"{status_text}")
    status_placeholder.progress((i + 1) / total_samples_in_pipeline, text=f"已处理样本: {i + 1} / {total_samples_in_pipeline}")

    correct = st.session_state.correct_predictions
    total = st.session_state.total_predictions
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    with accuracy_placeholder.container():
        st.metric(label="**当前管道准确率**", value=f"{accuracy:.2f} %", delta=f"{correct} / {total} 正确")

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
    
    # 控制模拟速度
    time.sleep(SIMULATION_INTERVAL_SECONDS)

# 循环结束后，确保最终状态可见
status_placeholder.header("🏁 **模拟完成!**")