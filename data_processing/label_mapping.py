def map_labels_in_dataset(dataset, label_mapping):
    """Applies the label mapping to a dataset."""
    def map_label(example):
        if example['label'] in label_mapping:
            example['label'] = label_mapping[example['label']]
        return example

    return dataset.map(map_label)
