def ensure_text_column(dataset):
    """Ensures the 'text' column exists by combining 'title' and 'body'."""
    if 'text' not in dataset.column_names:
        if 'title' in dataset.column_names and 'body' in dataset.column_names:
            def create_text(example):
                title = example['title'] if example['title'] is not None else ''
                body = example['body'] if example['body'] is not None else ''
                return {'text': title + ' ' + body}

            dataset = dataset.map(create_text)
        else:
            raise ValueError("Dataset does not contain 'text', 'title', or 'body' columns.")

    return dataset
