from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer as RobertaTrainer, TrainingArguments as RobertaTrainingArguments, RobertaConfig, DataCollatorWithPadding, TextClassificationPipeline
from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitTrainingArguments
from datetime import datetime
import torch


def train_setfit_model(config, base_model, output_path, train_set):
    model = SetFitModel.from_pretrained(base_model)

    args = SetFitTrainingArguments(
        output_dir=output_path,
        save_strategy="no",
        seed=config['random_seed'],
        batch_size=(16, 2),
        num_epochs=1,
        num_iterations=20,
    )

    trainer = SetFitTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time for {base_model}: {training_time}")

    return model, training_time

def train_roberta_model(config, base_model, output_path, train_set, label_to_int):
    MAX_LENGTH = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRUNCATION = True
    PADDING = 'max_length'

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=TRUNCATION, padding=PADDING, max_length=MAX_LENGTH)

    num_labels = len(train_set.unique('label'))
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    config = RobertaConfig.from_pretrained(base_model, num_labels=num_labels)
    model = RobertaForSequenceClassification(config=config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=DEVICE, truncation=TRUNCATION, padding=PADDING, max_length=MAX_LENGTH)

    train_set = train_set.map(preprocess_function, batched=True)
    train_set = train_set.map(lambda examples: {'label': label_to_int[examples['label']]})
    train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    args = RobertaTrainingArguments(
        output_dir=output_path,
        save_strategy="no",
        per_device_train_batch_size=16,
        num_train_epochs=10,
    )

    trainer = RobertaTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time for {base_model}: {training_time}")

    return classifier, training_time
