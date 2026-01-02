
HS_SEARCH_SPACE = {
    # Capacidad (ampliado pero razonable en 6GB)
    "d_model": {"type": "choice", "values": [256, 320, 384, 448, 512,768, 1024], "bw": 1},
    "ffn_hidden": {"type": "choice", "values": [512, 768, 1024, 1536, 2048], "bw": 1},
    "num_heads": {"type": "choice", "values": [2, 4, 8], "bw": 1},
    "num_layers": {"type": "choice", "values": [2, 3, 4, 5], "bw": 1},

    # Regularización
    "drop_prob": {"type": "float_range", "low": 0.0, "high": 0.12, "bw": 0.01},
    "label_smoothing": {"type": "choice", "values": [0.0, 0.05, 0.1, 0.15], "bw": 1},
    "weight_decay": {"type": "choice", "values": [0.0, 0.005, 0.01, 0.02], "bw": 1},

    # Optimización
    "lr": {"type": "choice", "values": [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4], "bw": 0},
    "warmup_steps": {"type": "choice", "values": [0, 100, 200, 400, 800], "bw": 1},
    "grad_clip": {"type": "choice", "values": [0.0, 0.5, 1.0, 2.0], "bw": 1},

    # Datos / batch (para 6GB: usar grad_accum para simular batch grande)
    "batch_size": {"type": "choice", "values": [2, 4, 8, 16], "bw": 1},
    "grad_accum_steps": {"type": "choice", "values": [1, 2, 4, 8], "bw": 1},

    "max_sequence_length": {"type": "choice", "values": [128, 160, 192, 224, 256, 288, 320], "bw": 1},

    # Entrenamiento
    "epochs": {"type": "choice", "values": [5], "bw": 1},
}
GA_SEARCH_SPACE = HS_SEARCH_SPACE

