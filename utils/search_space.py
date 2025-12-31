"""HS_SEARCH_SPACE = {
    "d_model": {"type": "choice", "values": [64, 128, 256, 512, 1024], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [128, 256, 512, 1024], "bw": 1},
    "num_heads": {"type": "choice", "values": [4, 8], "bw": 1},
    "drop_prob": {"type": "float_range", "low": 0.0, "high": 0.15, "bw": 0.05},
    "num_layers": {"type": "choice", "values": [2, 3, 4], "bw": 1},
    "lr": {"type": "choice", "values": [1e-4, 1e-3, 1e-2], "bw": 0},
    "batch_size": {"type": "choice", "values": [2, 4, 8, 16, 32, 64], "bw": 1},
    "max_sequence_length": {"type": "choice", "values": [100,150,200, 250,300], "bw": 1},
    "epochs": {"type": "choice", "values": [1, 2, 5], "bw": 1},
}"""

HS_SEARCH_SPACE = {
    "d_model": {"type": "choice", "values": [128, 256, 512], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [256, 512, 768], "bw": 1},
    "num_heads": {"type": "choice", "values": [4, 8], "bw": 1},
    "drop_prob": {"type": "float_range", "low": 0.0, "high": 0.2, "bw": 0.05},
    "num_layers": {"type": "choice", "values": [2, 3], "bw": 1},
    "lr": {"type": "choice", "values": [1e-4, 5e-4, 1e-3], "bw": 0},
    "batch_size": {"type": "choice", "values": [2, 4, 8], "bw": 1},
    "max_sequence_length": {"type": "choice", "values": [150, 200, 250, 300], "bw": 1},
    "epochs": {"type": "choice", "values": [1, 2, 5], "bw": 1},
}



GA_SEARCH_SPACE = {
    "d_model": {"type": "choice", "values": [64, 128, 256, 512, 1024], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [128, 256, 512, 1024], "bw": 1},
    "num_heads": {"type": "choice", "values": [4, 8], "bw": 1},
    "drop_prob": {"type": "float_range", "low": 0.0, "high": 0.15, "bw": 0.05},
    "num_layers": {"type": "choice", "values": [2, 3, 4], "bw": 1},
    "lr": {"type": "choice", "values": [1e-4, 1e-3, 1e-2], "bw": 0},
    "batch_size": {"type": "choice", "values": [2, 4, 8, 16, 32, 64], "bw": 1},
    "max_sequence_length": {"type": "choice", "values": [200], "bw": 1},
    "epochs": {"type": "choice", "values": [1, 2, 5], "bw": 1},
}
"""
GA_SEARCH_SPACE = {
    "d_model": {"type": "choice", "values": [64, 128, 256, 512], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [128,256, 512, 1024], "bw": 1},
    "num_heads": {"type": "choice", "values": [4,8], "bw": 1},
    "drop_prob": {"type": "float_range", "low": 0.0, "high": 0.15, "bw": 0.05},
    "num_layers": {"type": "choice", "values": [2, 3], "bw": 1},
    "lr": {"type": "choice", "values": [1e-4, 1e-3], "bw": 0},
    "batch_size": {"type": "choice", "values": [4, 8, 16], "bw": 1},
    "max_sequence_length": {"type": "choice", "values": [200], "bw": 1},
    "epochs": {"type": "choice", "values": [1, 2], "bw": 1},
}"""
