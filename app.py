import streamlit as st
import torch
import time
import os
import glob
import random
import pandas as pd
import altair as alt

# 确保导入了 dataset_config
from config import config, model_config, dataset_config 
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference

# 页面配置
st.set_page_config(page_title="多管道滑窗模拟 (v7)", layout="centered")

MODEL_PATH = "FFTbest_classification_model.pth"
CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]
NUM_PIPELINES = 100 # 最多模拟的管道数量
SIMULATION_INTERVAL_SECONDS = 0.01 # 每次滑窗处理之间停顿时间，0为无停顿，可以调大观察效果

@st.cache_resource
def load_resources():
    """
    加载模型和准备数据集，使用 @st.cache_resource 缓存，避免每次页面刷新都重新加载。
    """
    # 获取数据相关的配置，这里统一从 dataset_config["Data"] 获取
    data_cfg = dataset_config["Data"]

    try:
        # 初始化模型推理器，传入正确的 data_config
        predictor = ModelInference(
            model_path=MODEL_PATH,
            config=config, # 主配置
            data_config=data_cfg, # 数据集配置
            model_config=model_config,
            class_names=CLASS_NAMES
        )
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None, None

    try:
        # 获取所有符合文件模式的CSV文件路径
        all_file_paths = sorted(glob.glob(os.path.join(data_cfg["data_dir"], data_cfg["file_pattern"])))
        if not all_file_paths:
            st.error(f"找不到文件: {data_cfg['file_pattern']} 在 {data_cfg['data_dir']} 中。请检查路径和文件模式。")
            return predictor, None

        # 随机选择或全部选择要模拟的管道
        paths_to_simulate = all_file_paths if len(all_file_paths) <= NUM_PIPELINES else random.sample(all_file_paths, NUM_PIPELINES)
        
        # 计算采样点数的窗口大小和步长
        # config["window_size"] 和 config["stride"] 是以秒为单位的，需要乘以频率转换为采样点数
        window_size_samples = int(config["freq"] * config["window_size"])
        stride_samples = int(config["freq"] * config["stride"])

        pipeline_info = []
        for path in paths_to_simulate:
            data_dir = os.path.dirname(path)
            file_pattern_single = os.path.basename(path) # 只处理单个文件，所以模式就是文件名

            # 初始化TimeSeriesDataset，传入采样点数的窗口大小和步长
            # 确保传入正确的 Normalization, delete_col, constant_col, target_col
            dataset = TimeSeriesDataset(
                data_dir=data_dir,
                file_pattern=file_pattern_single, # 传递单个文件名作为模式
                window_size=window_size_samples, # 使用计算出的采样点数
                stride=stride_samples,       # 使用计算出的采样点数
                target_col=data_cfg["target_col"],
                Normalization=data_cfg.get("Normalization", True),
                delete_col=data_cfg["delete_col"],
                constant_col=data_cfg["constant_col"],
                # freq 参数不需要传递给 dataset，因为 window_size 和 stride 已经转换为采样点数
            )

            # 确保数据集中有样本
            if len(dataset) > 0:
                # 获取第一个样本的真实标签（假设所有样本的标签相同）
                _, label_tensor = dataset[0] 
                df_raw = pd.read_csv(path)
                pipeline_info.append({
                    'path': file_pattern_single,
                    'dataset': dataset,
                    'num_samples': len(dataset),
                    'true_label_idx': label_tensor.item(),
                    'raw_df': df_raw.reset_index(drop=True)
                })
            else:
                st.warning(f"文件 '{file_pattern_single}' 中没有有效样本生成，已跳过。")


        if not pipeline_info:
            st.error("无法从选定的文件中生成任何有效样本。请检查数据。")
            return predictor, None

        return predictor, pipeline_info

    except Exception as e:
        st.error(f"数据处理出错: {e}")
        return predictor, None

# 主逻辑开始
predictor, pipeline_info = load_resources()
if not predictor or not pipeline_info:
    st.stop() # 如果加载失败，停止执行

# 获取特征列名称，用于绘图
# 这些列在 ModelInference 中已经计算好，可以直接从 predictor 对象中获取
feature_cols_for_plot = predictor.feature_cols_names

# 会话状态，用于存储跨 reruns 的数据
if 'selected_pipeline_index' not in st.session_state:
    st.session_state.selected_pipeline_index = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# 侧边栏：管道选择
with st.sidebar:
    st.header("切换管道")
    # 遍历所有加载的管道信息，创建按钮
    for i, info in enumerate(pipeline_info):
        # 显示文件名的最后一部分，更简洁
        display_name = info['path'].split('_')[-1].replace('.csv', '') 
        name = f"管道 {i+1} ({display_name})"
        if st.button(name, key=f"pipe_{i}"):
            st.session_state.selected_pipeline_index = i
            st.session_state.correct_predictions = 0 # 切换管道时重置统计
            st.session_state.total_predictions = 0
            st.rerun() # 重新运行应用以加载新管道的数据

# 获取当前选定的管道信息
idx = st.session_state.selected_pipeline_index
current = pipeline_info[idx]
dataset = current["dataset"]
true_label = CLASS_NAMES[current["true_label_idx"]]
total_samples = current["num_samples"]
df_raw = current["raw_df"]

# 从配置中获取以采样点为单位的窗口大小和步长，用于计算绘图范围
window_size_samples = int(config["freq"] * config["window_size"])
stride_samples = int(config["freq"] * config["stride"])

# 主界面信息
st.title(f"当前管道: {current['path']}")
st.markdown(f"**真实标签:** `{true_label}` | **总样本数 (滑窗):** `{total_samples}`")
st.markdown("---")

# 创建占位符，以便在循环中动态更新UI元素
status_placeholder = st.empty()
accuracy_placeholder = st.empty()
result_placeholder = st.empty()
plot_placeholder = st.empty()

# 模拟主循环：遍历数据集中的每个滑窗样本
for i, (inputs_tensor, _) in enumerate(dataset):
    # 模型预测
    # inputs_tensor 的形状是 [window_size, num_features]
    # 模型期望的输入是 [batch_size, window_size, num_features]，所以需要 unsqueeze(0)
    inputs_tensor = inputs_tensor.unsqueeze(0).to(predictor.device) 
    
    with torch.no_grad(): # 推理模式，无需计算梯度
        outputs = predictor.model(inputs_tensor)

    # 获取预测结果
    _, predicted_idx = torch.max(outputs.data, 1)
    predicted_label = predictor.class_names[predicted_idx.item()]

    # 更新统计数据
    st.session_state.total_predictions += 1
    if predicted_label == true_label:
        st.session_state.correct_predictions += 1

    # 更新界面状态
    status_text = "🟢 正在模拟..." if i < total_samples - 1 else "🏁 模拟完成"
    status_placeholder.header(status_text)
    status_placeholder.progress((i + 1) / total_samples, text=f"处理样本: {i + 1} / {total_samples}")

    # 显示当前准确率
    acc = st.session_state.correct_predictions / st.session_state.total_predictions * 100
    with accuracy_placeholder.container():
        st.metric("当前准确率", f"{acc:.2f} %", f"{st.session_state.correct_predictions} / {st.session_state.total_predictions}")

    # 显示当前预测结果
    with result_placeholder.container():
        st.subheader("当前预测")
        col1, col2 = st.columns(2)
        with col1:
            st.write("预测")
            if predicted_label == true_label:
                st.success(f"✔️ {predicted_label}")
            else:
                st.error(f"❌ {predicted_label}") # 使用 st.error 表示错误预测
        with col2:
            st.write("真实")
            st.info(f"ℹ️ {true_label}")

    # 展示原始数据 ±2 窗口内容，用于上下文查看
    # 当前窗口的起始采样点
    current_window_start_idx = i * stride_samples 
    # 扩大范围：当前窗口前2个窗口到后2个窗口
    plot_start_idx = current_window_start_idx - 2 * window_size_samples
    plot_end_idx = current_window_start_idx + 3 * window_size_samples # 3 是为了包含当前窗口和其后2个窗口

    # 确保索引不越界
    plot_start_idx = max(0, plot_start_idx)
    plot_end_idx = min(len(df_raw), plot_end_idx)

    # 截取用于绘图的DataFrame
    df_plot = df_raw.iloc[plot_start_idx:plot_end_idx].copy()
    # 为Altair图表添加一个时间步（或索引）列
    df_plot["Timestep"] = df_plot.index # 使用原始DataFrame的索引作为时间步

    with plot_placeholder.container():
        st.subheader("传感器特征 (当前窗口及附近数据)")
        for feat in feature_cols_for_plot:
            if feat not in df_plot.columns: # 确保特征列存在于当前DataFrame中
                continue
            
            # 为每个特征创建图表
            # 使用 melt 操作将多列转换为长格式，方便 Altair 绘图
            # 注意: 如果只绘制一个特征，直接传递即可
            
            # 创建一个临时的DataFrame用于当前特征的绘图
            df_feat_plot = df_plot[[feat]].copy()
            df_feat_plot.index = df_plot["Timestep"] # 将Timestep设置为索引以便重置
            df_feat_plot = df_feat_plot.reset_index().rename(columns={feat: "Value"})

            chart = alt.Chart(df_feat_plot).mark_line().encode(
                x=alt.X("Timestep:Q", title="时间步 (采样点)"), # 明确是采样点
                y=alt.Y("Value:Q", title="数值")
            ).properties(
                title=f"特征: {feat}",
            ).interactive() # 添加交互性，如缩放和拖动
            st.altair_chart(chart, use_container_width=True) # 让图表宽度自适应容器

    # 每次滑窗处理后暂停，模拟实时处理
    time.sleep(SIMULATION_INTERVAL_SECONDS)

# 模拟结束后，最终状态
status_placeholder.header("🏁 模拟结束")
st.success(f"所有样本处理完毕！最终准确率：{acc:.2f} %")