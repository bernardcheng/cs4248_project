import numpy as np
import pandas as pd
from utils.formats import load_hdf
from typing import Tuple

import torch
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, RemBertForSequenceClassification, TrainingArguments, Trainer

MODEL_NAME = "google/rembert"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = RemBertForSequenceClassification.from_pretrained(MODEL_NAME)
VOCAB = TOKENIZER.get_vocab()
METRIC = evaluate.load("accuracy")

def process_input_embedding(file_path:str) -> Tuple[np.ndarray, pd.DataFrame]:

    input_embedding = load_hdf("data/conceptnet_api/retrofit/retrofitted-rembert-256")
    input_embedding_matrix = input_embedding.to_numpy()
    input_embedding_df = input_embedding.reset_index()
    input_embedding_df['vocab'] = input_embedding_df['index'].str.extract(r'/c/en/(\w+)/?')

    return input_embedding_matrix, input_embedding_df

def get_default_model_embeddings(model=MODEL) -> np.ndarray:

    rembert_model = model._modules['rembert']
    embedding_layer = rembert_model.embeddings.word_embeddings

    # torch.no_grad() to avoid tracking gradients
    with torch.no_grad():
        embedding_matrix = embedding_layer.weight.clone() # Clone to avoid modifying original

    default_embedding_matrix = embedding_matrix.cpu().numpy()
    return default_embedding_matrix

def modify_embedding(default_embedding_matrix:np.ndarray, input_embedding_matrix:np.ndarray, input_embedding_df:pd.DataFrame) -> Tuple[torch.tensor, dict]:

    modified_words = input_embedding_df['vocab'].to_list()

    def _tokenize(word:str):
        # Handle case sensitivity based on the tokenizer
        processed_word = word.lower() if TOKENIZER.do_lower_case else word

        # Tokenize the word - it might split into subwords
        tokens = TOKENIZER.tokenize(processed_word)
        return tokens

    modification_cache = dict() # store idx and words that were modified. 
    for idx, word in enumerate(modified_words):

        tokens = _tokenize(word)

        if len(tokens) == 1:

            token = tokens[0]
            embedding_idx = VOCAB[token]
            modification_cache['/c/en/' + word] = embedding_idx
            new_embedding_array = input_embedding_matrix[idx]
            default_embedding_matrix[embedding_idx] = new_embedding_array

    # Convert to PyTorch/TensorFlow tensor
    new_embedding_tensor = torch.tensor(default_embedding_matrix, dtype=torch.float16)
    return new_embedding_tensor, modification_cache

def replace_model_embeddings(new_embedding_tensor:torch.tensor, model=MODEL) -> None:

    rembert_model = model._modules['rembert']
    embedding_layer = rembert_model.embeddings.word_embeddings

    # Replace the weights (ensure device placement is correct if using GPU)
    with torch.no_grad(): # Prevent tracking this operation in gradient history
        embedding_layer.weight.copy_(new_embedding_tensor) # In-place copy is safer

    # Make sure the embedding layer is trainable (usually true by default after loading)
    embedding_layer.weight.requires_grad = True

def tokenize_function(examples):
    """
    Tokenize the `base_sentence` column so that it can be used as input to finetune REMBERT
    """
    return TOKENIZER(examples["base_sentence"], padding="max_length", truncation=True, max_length=512)

def prep_train_test_dataset() -> Tuple[Dataset, Dataset]:

    df = pd.read_csv("hf://datasets/flax-sentence-embeddings/Gender_Bias_Evaluation_Set/bias_evaluation.csv")
    df['labels'] = df['stereotypical_gender'].apply(lambda x: 1 if x == "male" else 0)
    # Convert dataset into Huggingface Dataset object with train-test split of 80:20
    datasets = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    train_dataset = datasets["train"]
    val_dataset = datasets["test"]

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = val_dataset.map(tokenize_function, batched=True)

    # Format the dataset for PyTorch - Remove columns not needed by the model
    cols_to_remove = ["Unnamed: 0", "base_sentence", "occupation", "male_sentence", "female_sentence", "stereotypical_gender"]
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(cols_to_remove)
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(cols_to_remove)

    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    return tokenized_train_dataset, tokenized_eval_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Logits are the raw output scores from the model, shape (batch_size, num_labels)
    # Labels are the ground truth, shape (batch_size,)
    predictions = np.argmax(logits, axis=-1)
    return METRIC.compute(predictions=predictions, references=labels)

def main(baseline=True):
    
    print(f"Starting {baseline}, cuda: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    input_embedding_matrix, input_embedding_df = process_input_embedding("data/conceptnet_api/retrofit/retrofitted-rembert-256")
    default_embedding_matrix = get_default_model_embeddings(model=MODEL)
    new_embedding_tensor, modification_cache = modify_embedding(default_embedding_matrix, input_embedding_matrix, input_embedding_df)
    
    if not baseline:
        replace_model_embeddings(new_embedding_tensor)

    tokenized_train_dataset, tokenized_eval_dataset = prep_train_test_dataset()
    print("Dataset prepared")
    training_args = TrainingArguments(
        output_dir="./results",             # Directory to save model checkpoints and logs
        num_train_epochs=3,                 # Reduced for quick demonstration; use more epochs (e.g., 3-5) for real tasks
        per_device_train_batch_size=8,      # Adjust based on your GPU memory
        per_device_eval_batch_size=8,       # Adjust based on your GPU memory
        warmup_steps=100,                   # Number of steps for linear warmup
        weight_decay=0.01,                  # Regularization strength
        logging_dir="./logs",               # Directory for TensorBoard logs
        logging_steps=50,                   # Log metrics every 50 steps
        # evaluation_strategy="epoch",        # Evaluate performance at the end of each epoch
        # save_strategy="epoch",              # Save model checkpoint at the end of each epoch
        # load_best_model_at_end=True,        # Load the best model found during training at the end
        metric_for_best_model="accuracy",   # Metric used to determine the best model
        greater_is_better=True,             # Accuracy should be maximized
        report_to="tensorboard",            # Report logs to TensorBoard (can add "wandb" etc.)
        # push_to_hub=False,                # Set to True to push model to Hugging Face Hub
        fp16=torch.cuda.is_available(),     # Use mixed precision training if CUDA is available
    )

    trainer = Trainer(
        model=MODEL,                        # The model to train (potentially with custom embeddings)
        args=training_args,                 # Training arguments defined above
        train_dataset=tokenized_train_dataset, # Training dataset
        eval_dataset=tokenized_eval_dataset,   # Evaluation dataset
        tokenizer=TOKENIZER,                # Tokenizer used for data collation (handles padding dynamically if needed)
        compute_metrics=compute_metrics,    # Function to compute evaluation metrics
        # Optional: Data collator can optimize padding
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    train_result = trainer.train()
    print("Training completed")

    trainer.save_model()  # Saves the tokenizer too
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Training data saved")

    # Evaluate the Final Model
    print("Evaluating the final model...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation Metrics: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Access the embedding layer again (use the same path as in Step 4)
    final_embedding_layer = MODEL.rembert.embeddings.word_embeddings

    # Get the weights
    final_embeddings_tensor = final_embedding_layer.weight.data

    # Convert to NumPy if desired (and move to CPU if on GPU)
    final_embeddings_numpy = final_embeddings_tensor.cpu().numpy()

    conceptnet_finetune_embeddings = dict()

    for concept, idx in modification_cache.items():
        conceptnet_finetune_embeddings[concept] = final_embeddings_numpy[idx].tolist()

    conceptnet_finetune_embeddings_df = pd.DataFrame.from_dict(conceptnet_finetune_embeddings, orient='index')
    
    if baseline:
        filepath = "finetuned_embeddings.hdf"
    else:
        filepath = "finetuned_custom_embeddings.hdf"

    conceptnet_finetune_embeddings_df.to_hdf(filepath, 'mat', encoding='utf-8')


if __name__ == "__main__":
    main(baseline=True)  
    main(baseline=False) 
