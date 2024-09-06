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

def preprocess_dataset(config):
    """Preprocesses the dataset according to the config."""
    ds = load_dataset(config['train_path'], config['test_path'], config['random_seed'])
    train_set = ds['train'] if 'train' in ds else None
    test_set = ds['test'] if 'test' in ds else None

    if train_set is None and test_set is None:
        raise ValueError("Both train_set and test_set are None. At least one must be provided for training or testing.")

    if config.get('map_labels', False):
        label_mapping = config.get('label_mapping', {})
        train_set = map_labels_in_dataset(train_set, label_mapping)
        if test_set:
            test_set = map_labels_in_dataset(test_set, label_mapping)

    if config.get('merge_train', False):
        add_ds = load_dataset(config['add_train'], None, config['random_seed'])['train']
        filtered_add_ds = filter_dataset_by_labels(add_ds, config['add_labels'])
        train_set = merge_datasets(train_set, filtered_add_ds)

    return train_set, test_set
