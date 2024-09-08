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

# save the input text together with the ground truth and predicted labels
def save_csv_results(test_set, references, predictions, output_path):
    """Saves the input text with ground truth and predicted labels."""
    output_file_name = 'results.csv'
    with open(os.path.join(output_path, output_file_name), 'w') as fp:
        fp.write("text,ground_truth,predicted\n")
        for i, (text, ref, pred) in enumerate(zip(test_set['text'], references, predictions)):
            fp.write(f"{text},{ref},{pred}\n")