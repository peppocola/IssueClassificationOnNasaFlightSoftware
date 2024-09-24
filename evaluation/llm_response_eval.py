import os
import json
import re
import yaml
from sklearn.metrics import classification_report
import pandas as pd
from openpyxl import Workbook
import argparse
import sys

sys.path.append('externals/sklearn-cls-report2excel')
from convert_report2excel import convert_report2excel

class LLMEvaluator:
    def __init__(self, config, single_model=None):
        self.config = config
        self.prompts_path = self.config["prompts_path"]
        self.responses_dir = self.config["responses_dir"]
        self.single_model = single_model
        self.possible_labels = self.get_possible_labels()

    def get_possible_labels(self):
        with open(self.prompts_path, 'r') as file:
            prompts = json.load(file)
        labels = set()
        for prompt_id, prompt_data in prompts.items():
            labels.add(prompt_data["target"])
        return list(labels)

    def get_response_paths(self):
        if self.single_model:
            return [os.path.join(self.responses_dir, self.single_model, self.config["response_file"])]
        
        response_paths = []
        for folder in os.listdir(self.responses_dir):
            folder_path = os.path.join(self.responses_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for subfolder in os.listdir(folder_path):
                response_path = os.path.join(folder_path, subfolder, "responses.json")
                response_paths.append(response_path)
        return response_paths

    def get_model_name(self, response_path):
        return response_path.split("/")[-2]

    def get_label(self, text):
        try:
            label = text['label']
            return label
        except:
            try:
                # labels should be the ones in the self.possible_labels list
                label = re.search(r"(:?\\\"|\")label(:?\\\"|\"):\s*(:?\\\"|\")(" + "|".join(self.possible_labels) + ")(:?\\\"|\")", text, flags=re.DOTALL)[4]
                return label
            except Exception:
                return ""

    def get_predictions(self, response_path):
        with open(response_path, 'r') as file:
            responses = json.load(file)
        predictions = {}
        for prompt_id, response in responses.items():
            predictions[prompt_id] = self.get_label(response)
        return predictions

    def get_true_labels(self):
        with open(self.prompts_path, 'r') as file:
            prompts = json.load(file)
        true_labels = {}
        for prompt_id, prompt_data in prompts.items():
            true_labels[prompt_id] = prompt_data["target"]
        return true_labels

    def evaluate_model(self):
        response_paths = self.get_response_paths()
        true_labels = self.get_true_labels()
        
        labels = list(set(true_labels.values()))

        metrics = {}
        for response_path in response_paths:
            model_name = self.get_model_name(response_path)
            if not os.path.exists(response_path):
                print(f"Warning: Response file not found for model {model_name}")
                continue
            predictions = self.get_predictions(response_path)
            y_true = []
            y_pred = []
            for prompt_id, true_label in true_labels.items():
                y_true.append(true_label)
                y_pred.append(predictions.get(prompt_id, ""))
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
            metrics[model_name] = report
        return metrics
