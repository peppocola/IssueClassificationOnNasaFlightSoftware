import os
import json
from sklearn.metrics import classification_report

def save_results(results, output_path, model_name):
    """Saves the classification report and metrics."""
    output_file_name = 'results.json'
    with open(os.path.join(output_path, output_file_name), 'w') as fp:
        json.dump(results, fp, indent=2)

def generate_classification_report(references, predictions):
    """Generates and returns a classification report."""
    return classification_report(references, predictions, digits=4, output_dict=True)
