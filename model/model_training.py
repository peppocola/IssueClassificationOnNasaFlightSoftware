import yaml
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer as RobertaTrainer, TrainingArguments as RobertaTrainingArguments, RobertaConfig, DataCollatorWithPadding, TextClassificationPipeline
from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitTrainingArguments
from datasets import Dataset
from datetime import datetime
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

roberta_config = load_config('config_roberta.yaml')
setfit_config = load_config('config_setfit.yaml')

class CustomRobertaTrainer(RobertaTrainer):
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    overall_recall = recall_score(labels, predictions, average='macro')
    class_recall = recall_score(labels, predictions, average=None)
    
    recall_dict = {f"recall_class_{i}": recall for i, recall in enumerate(class_recall)}
    recall_dict["recall_macro"] = overall_recall
    
    return recall_dict

def split_dataset(dataset, validation_split):
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=validation_split,
        random_state=roberta_config['random_seed']
    )
    return dataset.select(train_idx), dataset.select(val_idx)

def train_setfit_model(config, train_set, val_data=None):
    if config['use_validation'] and val_data is None:
        train_set, val_data = split_dataset(train_set, config['validation_split'])

    model = SetFitModel.from_pretrained(config['base_model'])

    args = SetFitTrainingArguments(
        output_dir=config['output_dir'],
        save_strategy=config['save_strategy'],
        seed=config['random_seed'],
        batch_size=tuple(config['batch_size']),
        num_epochs=config['num_epochs'],
        num_iterations=config['num_iterations'],
    )

    trainer = SetFitTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_data if config['use_validation'] else None,
        compute_metrics=compute_metrics
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time for {config['base_model']}: {training_time}")

    return model, training_time

def train_roberta_model(config, train_set, label_to_int, val_data=None):
    DEVICE = torch.device(config['device'])

    if config['use_validation'] and val_data is None:
        train_set, val_data = split_dataset(train_set, config['validation_split'])

    if config['use_custom_loss']:
        class_weights = torch.tensor(config['class_weights']).to(DEVICE)
        custom_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        custom_loss = None

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=config['truncation'], padding=config['padding'], max_length=config['max_length'])

    tokenizer = RobertaTokenizer.from_pretrained(config['base_model'])
    model_config = RobertaConfig.from_pretrained(config['base_model'], num_labels=config['num_labels'])
    model = RobertaForSequenceClassification(config=model_config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=DEVICE, 
                                            truncation=config['truncation'], padding=config['padding'], 
                                            max_length=config['max_length'])

    train_set = train_set.map(preprocess_function, batched=True)
    train_set = train_set.map(lambda examples: {'label': label_to_int[examples['label']]})
    train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    if val_data:
        val_data = val_data.map(preprocess_function, batched=True)
        val_data = val_data.map(lambda examples: {'label': label_to_int[examples['label']]})
        val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = RobertaTrainingArguments(
        output_dir=config['output_dir'],
        save_strategy=config['save_strategy'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        num_train_epochs=config['num_train_epochs'],
        evaluation_strategy=config['evaluation_strategy'] if config['use_validation'] else "no",
        logging_dir=config['logging_dir'],
        load_best_model_at_end=config['load_best_model_at_end'] if config['use_validation'] else False,
        metric_for_best_model=config['metric_for_best_model'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    if config['use_custom_loss']:
        trainer = CustomRobertaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_data if config['use_validation'] else None,
            data_collator=data_collator,
            tokenizer=tokenizer,
            custom_loss=custom_loss,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = RobertaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_data if config['use_validation'] else None,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time for {config['base_model']}: {training_time}")

    return classifier, training_time
