import os
from config.config_loader import load_config
from data_processing.dataset_utils import preprocess_dataset, print_label_distribution
from model.model_training import train_setfit_model, train_roberta_model
from model.model_prediction import predict_setfit, predict_roberta
from evaluation.result_saving import save_results, generate_classification_report
from setfit import SetFitModel

def process_model(config, model_type, train_set, test_set):
    """Process a single model for training or prediction."""
    output_path = os.path.join(config['output_path'], config['base_model'].split('/')[-1])
    os.makedirs(output_path, exist_ok=True)

    predict_mapping = config.get('label_to_int', {})

    if config.get('just_predict', False):
        if model_type == 'setfit':
            model = SetFitModel.from_pretrained(config['base_model'])
            references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
        else:
            raise ValueError("'just_predict' can only be used with the 'setfit' model type.")
    else:
        if model_type == 'setfit':
            model, training_time = train_setfit_model(config, train_set)
            references, predictions, prediction_time = predict_setfit(model, test_set, predict_mapping)
        elif model_type == 'roberta':
            model, training_time = train_roberta_model(config, train_set, predict_mapping, val_data=test_set)
            references, predictions, prediction_time = predict_roberta(model, test_set, predict_mapping)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        training_time_sec = training_time.total_seconds()

    results = generate_classification_report(references, predictions)
    results['prediction_time'] = prediction_time.total_seconds()
    if not config.get('just_predict', False):
        results['training_time'] = training_time_sec

    save_results(results, output_path, config['base_model'])
    print(f"Results for {config['base_model']}:")
    print(results)

def main():
    main_config = load_config("config.yaml")
    if not main_config:
        return

    model_type = main_config.get('model_type', 'setfit')

    if main_config.get('just_predict', False) and model_type != 'setfit':
        raise ValueError("'just_predict' can only be used with the 'setfit' model type.")

    # Load model-specific config
    model_config = load_config(f"config_{model_type}.yaml")
    if not model_config:
        return

    # Merge main config with model-specific config, prioritizing model-specific settings
    config = {**main_config, **model_config}

    # Prepare datasets
    train_set, test_set = preprocess_dataset(config)

    print("Training set label distribution:")
    print_label_distribution(train_set)

    if test_set:
        print("Test set label distribution:")
        print_label_distribution(test_set)

    # Process the model
    process_model(config, model_type, train_set, test_set)

if __name__ == "__main__":
    main()