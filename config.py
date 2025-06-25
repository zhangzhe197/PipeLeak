config = {
        "data_dir": "/home/zhangzhe/data/leak/processed_data",
        "file_pattern": 'leak_exp_*.csv',
        "window_size": int(2560 * 0.1),
        "stride": int(2560 * 0.1), 
        "target_col": 'LeakType',
        "batch_size": 64,
        "num_workers": 8,  # 根据你的CPU核心数调整
        "validation_split": 0.2, # 使用20%的数据作为验证集
        
        "hidden_size": 256,
        "num_layers": 4,
        "output_size": 4,   # 类别数量，对应 LeakType 的不同种类
        "dropout": 0.2,
        
        "Normalization": False,  # 是否启用归一化
        "model": "Transformer",  # 可选 "LSTM" 或 "Transformer"
        "num_epochs": 50,   # 增加训练轮数以观察效果
        "learning_rate": 0.00001,

        
        "model_save_path": "best_classification_model_TRANS.pth"
    }

model_config = {
        "LSTM": {        
            "hidden_size": 256,
            "num_layers": 4,
            "output_size": 4,   # 类别数量，对应 LeakType 的不同种类
            "dropout": 0.2,
        }, 

        "Transformer": {
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.2,
            "output_size": 4,   # 类别数量，对应 LeakType 的不同种类
        }
    }