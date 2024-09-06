from datetime import datetime
import torch

def predict_setfit(model, test_set, predict_mapping):
    """Predicts labels for the test set and maps the predictions."""
    references = list(test_set['label'])

    start_time = datetime.now()
    predictions_raw = list(model.predict(test_set['text'], batch_size=8, show_progress_bar=True))
    end_time = datetime.now()

    prediction_time = end_time - start_time
    print(f"Prediction time: {prediction_time}")

    # Map the predictions based on their format
    predictions = map_predictions(predictions_raw, predict_mapping)

    if len(predictions) != len(references):
        print(f"Warning: Number of references ({len(references)}) does not match number of predictions ({len(predictions)}).")

    return references, predictions, prediction_time


def predict_roberta(classifier, test_set, predict_mapping):
    """Predicts labels for the test set and maps the predictions."""
    references = list(test_set['label'])

    start_time = datetime.now()
    predictions_raw = classifier(test_set['text'])
    predictions_raw = [item['label'] for item in predictions_raw]
    end_time = datetime.now()

    prediction_time = end_time - start_time
    print(f"Prediction time: {prediction_time}")

    # Map the predictions based on their format
    predictions = map_predictions(predictions_raw, predict_mapping)

    if len(predictions) != len(references):
        print(f"Warning: Number of references ({len(references)}) does not match number of predictions ({len(predictions)}).")

    return references, predictions, prediction_time


def map_predictions(predictions_raw, predict_mapping):
    inverted_mapping = {v: k for k, v in predict_mapping.items()}
    predictions = []
    for item in predictions_raw:
        if isinstance(item, dict) and 'label' in item:
            # Extract label from dictionary
            item = item['label']
        elif isinstance(item, torch.Tensor):
            # Extract integer from tensor
            item = item.item()
            
        if isinstance(item, int):
            # Map integer prediction
            predictions.append(inverted_mapping[predict_mapping.get(item, item)])
        elif isinstance(item, str) and item.startswith('LABEL_'):
            # Extract integer from string and map
            try:
                item = int(item.split('_')[1])
                predictions.append(inverted_mapping[predict_mapping.get(item, item)])
            except ValueError as e:
                print(f"Warning: Could not convert prediction '{item}' to integer. Error: {e}")
        else:
            # Keep the prediction as-is if it doesn't match other formats
            predictions.append(item)

    return predictions
