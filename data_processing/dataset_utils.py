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


def load_dataset(train_path, test_path=None, random_seed=42):
    """Loads and shuffles train and test datasets."""
    data_files = {"train": train_path}
    if test_path:
        data_files["test"] = test_path

    ds = Dataset.from_csv(data_files)
    ds = ds.shuffle(seed=random_seed)
    return ds

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

    # Merge text columns if specified and merged_text_column doesn't exist in both sets
    if 'text_columns' in config:
        if (train_set is None or config['merged_text_column'] not in train_set.column_names) and \
           (test_set is None or config['merged_text_column'] not in test_set.column_names):
            def default_merge(texts):
                return '\n'.join(str(text) for text in texts if text)
            
            merge_func = merge_text_func or default_merge
            
            if train_set is not None:
                train_set = merge_text_columns(train_set, config['text_columns'], config['merged_text_column'], merge_func)
            
            if test_set is not None:
                test_set = merge_text_columns(test_set, config['text_columns'], config['merged_text_column'], merge_func)

    # Ensure 'text' and 'label' columns are present
    if train_set is not None:
        train_set = ensure_required_columns(train_set, config)
    
    if test_set is not None:
        test_set = ensure_required_columns(test_set, config)

    return train_set, test_set

def merge_text_columns(dataset, text_columns, merged_column, merge_func):
    merged_texts = [merge_func([dataset[col][i] for col in text_columns]) for i in range(len(dataset))]
    dataset = dataset.add_column(merged_column, merged_texts)
    dataset = dataset.remove_columns(text_columns)
    return dataset

def ensure_required_columns(dataset, config):
    if 'text' not in dataset.column_names and config['merged_text_column'] in dataset.column_names:
        dataset = dataset.rename_column(config['merged_text_column'], 'text')
    if 'label' not in dataset.column_names and 'label' in config.get('label_mapping', {}):
        dataset = dataset.rename_column(config['label_mapping']['label'], 'label')
    return dataset
