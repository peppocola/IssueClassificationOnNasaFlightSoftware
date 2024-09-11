import pandas as pd
from datasets import Dataset as HFDataset
from typing import Union, Dict, List, Any, Callable, Optional
from tqdm import tqdm
import yaml
import json

class PromptGenerator:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.params = yaml.safe_load(file)
        
        self.dataset = self.load_dataset(self.params['data_path'])
        self.template = self.load_prompt_template(self.params['template_path'])
        self.config = self.load_llm_config(self.params['config_llm_path'])
        self.prompts = {}

    def load_dataset(self, data_path: str) -> Union[pd.DataFrame, HFDataset]:
        """
        Load a dataset from a file or Hugging Face dataset name.

        Args:
            data_path (str): Path to the dataset file or name of the Hugging Face dataset.

        Returns:
            Union[pd.DataFrame, HFDataset]: The loaded dataset.
        """
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            # Assume it's a Hugging Face dataset name
            return hf_load_dataset(data_path)

    def load_prompt_template(self, template_path: str) -> Dict[str, Any]:
        """
        Load the prompt template from a YAML file.

        Args:
            template_path (str): Path to the YAML file containing the prompt template.

        Returns:
            Dict[str, Any]: The loaded prompt template.
        """
        with open(template_path, 'r') as file:
            return yaml.safe_load(file)

    def load_llm_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the LLM configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML file containing the LLM configuration.

        Returns:
            Dict[str, Any]: The loaded LLM configuration.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def clean_prompt_template(self) -> Dict[str, Any]:
        """
        Clean the prompt template by removing unused fields based on the configuration.

        Args:
            template (Dict[str, Any]): The original prompt template.
            config (Dict[str, Any]): The configuration dictionary.

        Returns:
            Dict[str, Any]: The cleaned prompt template.
        """
        cleaned_template = self.template.copy()
        
        # Remove system message if not used
        if not self.config['use_system_message']:
            cleaned_template.pop('system', None)
        
        # Remove examples if few-shot prompting is not used
        if not self.config['few_shot_prompt']:
            cleaned_template.pop('examples', None)
        
        return cleaned_template

    def dataset_iterator(self):
        """
        Create an iterator for either a pandas DataFrame or a Hugging Face Dataset.

        Args:
            dataset (Union[pd.DataFrame, HFDataset]): The input dataset.

        Returns:
            An iterator over the dataset.
        """
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.iterrows()
        elif isinstance(self.dataset, HFDataset):
            return enumerate(self.dataset)
        else:
            raise ValueError("Unsupported dataset type. Use either pandas DataFrame or Hugging Face Dataset.")

    def select_few_shot_examples(self, num_samples_per_class: int, random_sampling: bool = True, sampling_function: Optional[Callable] = None, seed: int = 42) -> list:
        """
        Select examples for few-shot prompting from the training dataset.

        Args:
        trainset_path (str): Path to the training dataset CSV file.
        num_samples_per_class (int): Number of samples to select for each class.
        random_sampling (bool): If True, use random sampling. If False, use the provided sampling function.
        sampling_function (Callable, optional): A function to use for sampling when random_sampling is False.
        seed (int): Seed for random number generator. Default is 42.

        Returns:
        list: A list of selected examples for few-shot prompting.
        """
        train_data = pd.read_csv(self.params['trainset_path'])
        unique_labels = train_data['label'].unique()
        selected_examples = []

        if random_sampling:
            # Set the seed for reproducibility
            pd.np.random.seed(seed)

        for label in unique_labels:
            class_data = train_data[train_data['label'] == label]
            
            if random_sampling:
                if len(class_data) >= num_samples_per_class:
                    selected_examples.extend(class_data.sample(n=num_samples_per_class).to_dict('records'))
                else:
                    selected_examples.extend(class_data.to_dict('records'))
                    print(f"Warning: Only {len(class_data)} samples available for class '{label}', which is less than the requested {num_samples_per_class}.")
            else:
                if sampling_function is None:
                    raise ValueError("A sampling function must be provided when random_sampling is False.")
                
                selected_examples.extend(sampling_function(class_data, num_samples_per_class))

        return selected_examples

    def craft_prompt(self, replacements: Dict[str, str]) -> str:
        """
        Craft a prompt by replacing placeholders in the template with provided values.

        Args:
            template (Dict[str, str]): The prompt template.
            replacements (Dict[str, str]): Dictionary of placeholder replacements.

        Returns:
            str: The crafted prompt.
        """
        example = self.replace_placeholders(self.template['example'], replacements)
        return '\n'.join([self.template['task'], self.template['label_explanations'], example, self.template['format_instructions'], self.template['output']])

    def replace_placeholders(self, template: str, replacements: Dict[str, str]) -> str:
        """
        Replace placeholders in a template string with values from the dataset.

        Args:
            template (str): The template string with placeholders.
            replacements (Dict[str, str]): Dictionary of placeholder replacements.

        Returns:
            str: The template with placeholders replaced.
        """
        
        return template.format(**replacements)

    def generate_prompts(self) -> None:
        cleaned_template = self.clean_prompt_template()
        
        for index, row in tqdm(self.dataset_iterator()):
            replacements = {column: row[column] for column in self.params['column_names']}
            example_to_classify = dict(row) if isinstance(self.dataset, HFDataset) else row.to_dict()
            crafted_prompt = self.craft_prompt(cleaned_template, replacements, self.config, self.params['trainset_path'], example_to_classify)
            test_pair = {
                'prompt': crafted_prompt['prompt'],
                'system': crafted_prompt['system'],
                'target': row[self.params['target_column']]
            }
            self.prompts[index] = test_pair

    def save_prompts(self) -> None:
        with open(self.params['prompts_path'], 'w') as file:
            json.dump(self.prompts, file, indent=4)

    def run(self):
        self.generate_prompts()
        self.save_prompts()

if __name__ == "__main__":
    generator = PromptGenerator()
    generator.run()
