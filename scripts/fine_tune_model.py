from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Load a pre-trained Vision-Encoder-Decoder model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load and preprocess the dataset
def preprocess_data(examples):
    pixel_values = feature_extractor(examples['image'], return_tensors="pt").pixel_values
    labels = tokenizer(examples['text'], padding="max_length", truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values.squeeze(), "labels": labels.squeeze()}

dataset = load_dataset("path/to/your/dataset")
processed_dataset = dataset.map(preprocess_data, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    output_dir="./results",
    logging_dir="./logs",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
)

# Fine-tune the model
trainer.train()