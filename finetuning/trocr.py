import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, IntervalStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.to(device)
try:
    model = torch.compile(model)
except:
    pass

class OCRImageDataset(Dataset):
    def __init__(self, image_dir, label_file, processor, max_target_length=64):
        df = pd.read_csv(label_file, names=["filename", "text"])
        df = df.dropna(subset=["filename", "text"])
        df["filename"] = df["filename"].astype(str)
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        df = df[df["filename"].map(lambda f: os.path.exists(os.path.join(image_dir, f)))]
        self.samples = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn = self.samples.loc[idx, "filename"]
        txt = str(self.samples.loc[idx, "text"])
        img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        enc = self.processor(images=img, return_tensors="pt")
        labels = self.processor.tokenizer(
            txt,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        labels = torch.where(labels == processor.tokenizer.pad_token_id, -100, labels)
        return {"pixel_values": enc.pixel_values.squeeze(0), "labels": labels}

class TrOCRDataCollator:
    def __call__(self, features):
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }

data_collator = TrOCRDataCollator()
image_dir = "/content/ta/train"
train_label_file = "/content/ta/train.txt"
test_label_file = "/content/ta/test.txt"

train_dataset = OCRImageDataset(image_dir, train_label_file, processor)
eval_dataset = OCRImageDataset(image_dir, test_label_file, processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-tamil-finetuned",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    save_strategy=IntervalStrategy.EPOCH,
    save_total_limit=2,
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=10,
    logging_dir="./logs",
    predict_with_generate=True,
    fp16=True if device.type == "cuda" else False,
    remove_unused_columns=False,
    report_to="none",
    eval_strategy="no",
    dataloader_num_workers=4
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./trocr-tamil-finetuned")
processor.save_pretrained("./trocr-tamil-finetuned")
