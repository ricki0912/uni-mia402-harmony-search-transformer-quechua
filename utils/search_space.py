HS_SEARCH_SPACE = {
    "d_model": {"type": "choice", "values": [256, 320, 384, 448, 512], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [512, 640, 768, 896, 1024], "bw": 1},
    "num_heads": {"type": "choice", "values": [4, 6, 8], "bw": 1},
    "drop_prob": {"type": "float_range", "low": 0.05, "high": 0.25, "bw": 0.05},
    "num_layers": {"type": "choice", "values": [2, 3, 4], "bw": 1},
    "lr": {"type": "float_range", "low": 5e-5, "high": 5e-4, "bw": 5e-5},
    "batch_size": {"type": "choice", "values": [4, 8, 12, 16], "bw": 1},
    "max_sequence_length": {"type": "choice", "values": [128, 160, 192, 224, 256, 288, 320], "bw": 1},
    "epochs": {"type": "choice", "values": [6, 10, 14, 18], "bw": 1},
}

GA_SEARCH_SPACE = HS_SEARCH_SPACE

