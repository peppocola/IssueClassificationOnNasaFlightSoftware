from datasets import Dataset, concatenate_datasets
from data_processing.text_processing import ensure_text_column
from data_processing.label_mapping import map_labels_in_dataset
from collections import Counter

def print_label_distribution(dataset):
    if 'label' not in dataset.column_names:
        print("Dataset does not contain a 'label' column.")
        return

    labels = dataset['label']
    label_counts = Counter(labels)

    print("Label Distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} examples")


def load_dataset(train_path=None, test_path=None, random_seed=42):
    """Loads and shuffles train and test datasets separately."""
    if train_path is None and test_path is None:
        raise ValueError("At least one of train_path or test_path must be provided.")

    train_ds = None
    test_ds = None

    if train_path:
        train_ds = Dataset.from_csv(train_path)

    if test_path:
        test_ds = Dataset.from_csv(test_path)

    return {"train": train_ds, "test": test_ds}

def filter_dataset_by_labels(ds, labels):
    """Filters dataset by specific labels."""
    return ds.filter(lambda x: x['label'] in labels)

def merge_datasets(ds1, ds2):
    """Ensures common columns and merges datasets."""
    ds1 = ensure_text_column(ds1)
    ds2 = ensure_text_column(ds2)

    common_columns = set(ds1.column_names).intersection(set(ds2.column_names))
    ds1 = ds1.select_columns(common_columns)
    ds2 = ds2.select_columns(common_columns)

    return concatenate_datasets([ds1, ds2])

def preprocess_dataset(config, merge_text_func=None):
    """Preprocesses the dataset according to the config."""
    ds = load_dataset(config['train_path'], config['test_path'], config['random_seed'])
    train_set = ds['train'] if 'train' in ds else None
    test_set = ds['test'] if 'test' in ds else None

    if train_set is None and test_set is None:
        raise ValueError("Both train_set and test_set are None. At least one must be provided for training or testing.")

    # Merge text columns if specified
    if 'text_columns' in config:
        def default_merge(texts):
            return '\n'.join(str(text) for text in texts if text)
        
        merge_func = merge_text_func or default_merge
        
        if train_set is not None and config['merged_text_column'] not in train_set.column_names:
            train_set = merge_text_columns(train_set, config['text_columns'], config['merged_text_column'], merge_func)
            train_set = ensure_required_columns(train_set, config)
        
        if test_set is not None and config['merged_text_column'] not in test_set.column_names:
            test_set = merge_text_columns(test_set, config['text_columns'], config['merged_text_column'], merge_func)
            test_set = ensure_required_columns(test_set, config)

    # Apply label mapping to both train and test sets
    if train_set is not None and config['label_mapping']:
        train_set = map_labels_in_dataset(train_set, config['label_mapping'])
    
    if test_set is not None and config['label_mapping']:
        test_set = map_labels_in_dataset(test_set, config['label_mapping'])

    return train_set, test_set

def merge_text_columns(dataset, text_columns, merged_column, merge_func):
    merged_texts = [merge_func([dataset[col][i] for col in text_columns]) for i in range(len(dataset))]
    dataset = dataset.add_column(merged_column, merged_texts)
    dataset = dataset.remove_columns(text_columns)
    return dataset

def ensure_required_columns(dataset, config):
    if 'text' not in dataset.column_names:
        if config['merged_text_column'] in dataset.column_names:
            dataset = dataset.rename_column(config['merged_text_column'], 'text')
        else:
            raise ValueError("Text column not found in dataset.")
    
    if 'label' not in dataset.column_names:
        raise ValueError("Label column not found in dataset.")
    
    return dataset
