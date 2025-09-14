# Training Script Usage
## Command Format
```bash
python train_topoqa.py \
    --graph_dir <path_to_graph_directory> \
    --train_label_file <path_to_train_labels.csv> \
    --val_label_file <path_to_val_labels.csv> \
    [--attention_head 8] \
    [--pooling_type "mean"] \
    [--batch_size 16] \
    [--learning_rate 0.005] \
    [--num_epochs 200] \
    [--accumulate_grad_batches 32] \
    [--seed 222] \
    [--save_dir "./experiments"]
```

## Data Files Description
### Label Files
- **`train.csv`**: Training set labels, containing sample IDs and corresponding labels
- **`val.csv`**: Validation set labels, containing sample IDs and corresponding labels

### Data Splitting
- The **training/validation split is directly adopted** from DProQA's original data partitioning
- Split files ('tain.csv' and 'val.csv') are identical to those used in DProQA

### Excluded Conformations
The following 20 specific conformations from DProQA are **excluded** from our dataset beacause DProQA does not provide corresponding labels for them:

**1bui target (10 conformations):**
1bui_123095
1bui_169037
1bui_197493
1bui_197792
1bui_293265
1bui_30467
1bui_342063
1bui_352665
1bui_53336
1bui_95049;
**1zy8 target (10 conformations):**
1zy8_103554
1zy8_111999
1zy8_27967
1zy8_38510
1zy8_45073
1zy8_52940
1zy8_53510
1zy8_93282
1zy8_95997
1zy8_99144.

- Reference: [DProQA 1bui.csv](https://github.com/jianlin-cheng/DProQA/blob/main/Data/Docking_decoy_set/lable_csv/1bui.csv) and [DProQA 1zy8.csv](https://github.com/jianlin-cheng/DProQA/blob/main/Data/Docking_decoy_set/lable_csv/1zy8.csv)


### File Format
Both CSV files should contain at least the following columns:
- `MODEL`: Unique identifier for each sample
- `dockq`: The dockq value, calculated by fnat, LRMS, IRMS.
- `capri`
