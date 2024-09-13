def map_labels_in_dataset(dataset, label_mapping):
    """Applies the label mapping to a dataset."""
    return dataset.map(lambda example: 
                            {
                                **example,
                                'label': label_mapping.get(example['label'],
                                example['label'])
                            }
                    )
