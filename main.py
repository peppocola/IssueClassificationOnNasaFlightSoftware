import os
from config.config_loader import load_config
from data_processing.dataset_utils import preprocess_dataset, print_label_distribution
from model.model_training import train_setfit_model, train_roberta_model
from model.model_prediction import predict_setfit, predict_roberta
from evaluation.result_saving import save_results, generate_classification_report
from setfit import SetFitModel  # Assuming this is imported somewhere in the original script

def process_model(base_model, config, model_type, train_set, test_set):
    """Process a single model for training or prediction."""
    output_path = os.path.join(config['output_path'], base_model.split('/')[-1])
    os.makedirs(output_path, exist_ok=True)

    predict_mapping = config.get('label_to_int', {})

    # Handle just prediction logic
    if config.get('just_predict', False):
        if model_type == 'setfit':
            model = SetFitModel.from_pretrained(base_model)
            references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
        else:
            raise ValueError("'just_predict' can only be used with the 'setfit' model type.")
    else:
        if model_type == 'setfit':
            model, training_time = train_setfit_model(config, base_model, output_path, train_set)
            references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
        elif model_type == 'roberta':
            model, training_time = train_roberta_model(config, base_model, output_path, train_set, label_to_int)
            references, predictions, prediction_time = predict_roberta(model, test_set, predict_mapping)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        training_time_sec = training_time.total_seconds()

    # Generate results and save
    results = generate_classification_report(references, predictions)
    results['prediction_time'] = prediction_time.total_seconds()
    if not config.get('just_predict', False):
        results['training_time'] = training_time_sec

    save_results(results, output_path, base_model)
    print(f"Results for {base_model}:")
    print(results)


def handle_model_type(model_type, base_models, config, train_set, test_set):
    """Handles processing for different model types."""
    if model_type == 'setfit':
        for base_model in base_models:
            print(f"Processing model: {base_model}")
            process_model(base_model, config, model_type, train_set, test_set)
    elif model_type == 'roberta':
        base_model = 'roberta-base'  # Assuming only one model for RoBERTa
        print(f"Processing model: {base_model}")
        process_model(base_model, config, model_type, train_set, test_set)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    config = load_config("config.yaml")
    if not config:
        return

    base_models = config['base_model'] if isinstance(config['base_model'], list) else [config['base_model']]
    model_type = config.get('model_type', 'setfit')  # Get model type from config

    # Validate 'just_predict' setting
    if config.get('just_predict', False) and model_type != 'setfit':
        raise ValueError("'just_predict' can only be used with the 'setfit' model type.")

    # Prepare datasets
    train_set, test_set = preprocess_dataset(config)

    print("Training set label distribution:")
    print_label_distribution(train_set)

    if test_set:
        print("Test set label distribution:")
        print_label_distribution(test_set)

    # Handle model processing based on the model type
    handle_model_type(model_type, base_models, config, train_set, test_set)


if __name__ == "__main__":
    main()
