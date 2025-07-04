config = {

        "freq" : 25600, 
        "alignment": "Data",  # Data or Model
        "window_size": 0.01,
        "stride": 0.01,
        "batch_size": 64,
        "num_workers": 8,  # 根据你的CPU核心数调整
        "dropout": 0.2,
        "model": "PairedTransformer",  # 可选 "LSTM", "Transformer", "FFT", "CNN", "PairedFFTTransformer", “PairedTransformer”
        "num_epochs": 100,   # 增加训练轮数以观察效果
        "learning_rate": 0.001,
        "seed" : 1234,
        "milestone": [33,66],  # 学习率调整的里程碑
        "gamma": 0.2,  # 学习率调整的衰减
        "model_save_path": "best_classification_model.pth"

    }

model_config = {
        "LSTM": {        
            "hidden_size": 256,
            "num_layers": 4,
            "output_size": 5,   # 类别数量，对应 LeakType 的不同种类
            "dropout": 0.2,
    
        }, 

        "Transformer": {
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "output_size": 5,   # 类别数量，对应 LeakType 的不同种类
        }, 
        "FFT": {
            "output_size": 5,   # 类别数量，对应 LeakType 的不同种类
        },

        "CNN":{
            "in_channels": 1,  # 输入通道数，单变量时间序列为1
            "output_size": 5,   # 类别数量，对应 LeakType 的不同种类
        },
        "PairedFFTTransformer": 
        {
            "output_size": 5, # 类别数量
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "mlp_hidden_dim": 256
        },
    }

dataset_config = {
    "Data":
    {
        "all_columns": ['ValueH1N', 'ValueH2N', 'ValueA2', 'TimeA1', 'ValueA1', 'ValueP2', 'ValueP1'], 
        "Normalization": "Sample",  # 是否启用归一化
        "delete_col": ["Structure", "FlowCondition"],  # 删除不需要的列
        "constant_col": [],  # 常量列不参与归一化
        "validation_split": 0.2, # 使用20%的数据作为验证集
        "target_col": "LeakType",  # 目标列
        "data_dir": "/home/zhangzhe/data/processed_leak_data/",
        "file_pattern": 'leak_exp_*.csv',
    },
    "Model":
    {
        "Normalization": "Sample",  # 是否启用归一化
        "delete_col": ["Structure", "FlowCondition"],  # 删除不需要的列
        "constant_col": [],  # 常量列不参与归一化
        "validation_split": 0.2, # 使用20%的数据作为验证集
        "target_col": "LeakType",  # 目标列
        "window_size_sec": 0.01,  # 窗口大小，单位为秒
        "total_duration_sec": 30,  # 假设每个文件的总
        "stride_sec": 0.01,  # 窗口滑动步长，单位为秒

        "data_dir": "/home/zhangzhe/data/processed_leak_data/",
        "file_pattern_A":'leak_exp_Nsound_*.csv',
        "file_pattern_B":'leak_exp_sound_*.csv',
    }
}