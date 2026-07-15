# WM-811K Data Flow

Shape notation: `N` = number of wafers loaded from a split; `B` = wafers in one DataLoader batch.

```mermaid
flowchart TD
    RAW["One raw waferMap<br/>shape: (H, W)<br/>dtype: uint8; values: 0, 1, 2"]
    RESIZED["One resized wafer<br/>shape: (64, 64)<br/>dtype: uint8; values: 0, 1, 2"]
    PARQUET["One Silver or Gold row<br/>shape: (4096,)<br/>dtype: uint8; values: 0, 1, 2"]
    SPLIT["Stratified split<br/>train / val / test rows"]
    TRAIN["train.parquet<br/>each wafer: (4096,)"]
    EVAL["val.parquet or test.parquet<br/>each wafer: (4096,)"]
    LOADER["After loading a split<br/>(N, 4096) -> (N, 1, 64, 64), float32<br/>one sample: (1, 64, 64)<br/>one batch: (B, 1, 64, 64)"]
    MODEL["Inside model.forward<br/>shape stays: (B, 1, 64, 64)<br/>values become: 0.0, 0.5, 1.0"]
    LOGITS["CNN or ResNet output<br/>shape: (B, 8)"]
    VALIDATION["test_validation.py<br/>checks one Parquet row: (4096,), values 0, 1, 2"]

    RAW -->|"nearest neighbor resize"| RESIZED
    RESIZED -->|"flatten"| PARQUET
    PARQUET --> SPLIT
    SPLIT --> TRAIN
    SPLIT --> EVAL
    TRAIN -->|"weighted sampling; optional rotate or flip"| LOADER
    EVAL -->|"no augmentation"| LOADER
    LOADER -->|"inside model.forward"| MODEL
    MODEL --> LOGITS
    PARQUET -.-> VALIDATION
```
