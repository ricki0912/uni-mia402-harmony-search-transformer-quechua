import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def partition_dataset(input_file: str, output_dir: str = None, parts: list = None):
    """
    Particiona un dataset en múltiples partes
    Args:
        input_file: Ruta al archivo Excel con el dataset
        output_dir: Directorio para guardar las particiones (opcional)
        parts: Lista de porcentajes (debe sumar 100)
    """
    # Configurar directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or f'data/partitions_{timestamp}'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parts = parts or [10] * 10 
    
    if sum(parts) != 100:
        raise ValueError(f"Los porcentajes deben sumar 100, suma actual: {sum(parts)}")

    df = pd.read_excel(input_file)
    total_rows = len(df)
    

    df_shuffled = df.sample(frac=1, random_state=42)
    
    start_idx = 0
    partition_stats = []
    
    for i, percentage in enumerate(parts, 1):
        size = int(total_rows * percentage / 100)
        end_idx = start_idx + size
        
        partition = df_shuffled.iloc[start_idx:end_idx].copy()
        
        filename = output_path / f"partition_{i:02d}_{percentage}pct.xlsx"
        partition.to_excel(filename, index=False)
        
        # Recopilar estadísticas
        stats = {
            'partition': i,
            'percentage': percentage,
            'rows': len(partition),
            'es_chars_avg': partition['es'].str.len().mean(),
            'qu_chars_avg': partition['qu'].str.len().mean()
        }
        partition_stats.append(stats)
        
        start_idx = end_idx
    
    # Guardar resumen de estadísticas
    stats_df = pd.DataFrame(partition_stats)
    stats_file = output_path / "partition_stats.xlsx"
    stats_df.to_excel(stats_file, index=False)
    
    return output_path

if __name__ == "__main__":
    # Configuración
    input_file = "data/data_preprocesada_sin_outliers_v4_biblia.xlsx"
    
    # Crear 10 particiones iguales
    output_dir = partition_dataset(input_file)
    
    # O crear particiones desiguales (10%, 20%, 30%, 40%)
    unequal_parts = [10, 20, 30, 40,50,60,70,80,90,100]
    output_dir = partition_dataset(input_file, 
                                 output_dir='data/unequal_partitions',
                                 parts=unequal_parts)
