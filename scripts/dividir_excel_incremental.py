import pandas as pd
from pathlib import Path

def split_excel_incremental(
    input_path: str,
    initial_chunk: int = 10_000,
    step: int = 10_000,
) -> None:
    src = Path(input_path)
    df = pd.read_excel(src)
    total_rows = len(df)

    offset = 0
    chunk_size = initial_chunk
    part = 1

    while offset < total_rows:
        end = min(offset + chunk_size, total_rows)
        chunk = df.iloc[offset:end]
        out_name = f"{src.stem}_part_{part}_{offset+1}-{end}.xlsx"
        chunk.to_excel(src.with_name(out_name), index=False)

        offset = end
        part += 1
        chunk_size += step  # 10k, 20k, 30k, ...

if __name__ == "__main__":
    split_excel_incremental("../data/data_preprocesada_sin_outliers_v4_biblia.xlsx")