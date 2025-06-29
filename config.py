config = {
        "data_dir": "/home/zhangzhe/data/leak/processed_data",
        "file_pattern": 'leak_exp_*.csv',
        "window_size": int(2560 * 0.1),
        "stride": int(2560 * 0.1), 
        "target_col": 'LeakType',
        "batch_size": 64,
        "num_workers": 1,  # 根据你的CPU核心数调整
        "validation_split": 0.2, # 使用20%的数据作为验证集
        "delete_col": ["Structure", "FlowCondition"],  # 删除不需要的列
        "constant_col": [],  # 常量列不参与归一化
        "dropout": 0.2,
        "all_columns": ['ValueH1N', 'ValueH2N', 'ValueA2', 'TimeA1', 'ValueA1', 'ValueP2', 'ValueP1'], 
        "Normalization": "Sample",  # 是否启用归一化
        "model": "FFT",  # 可选 "LSTM" 或 "Transformer"
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
        }
    }