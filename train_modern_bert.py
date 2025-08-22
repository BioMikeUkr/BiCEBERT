from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed, ModernBertConfig, ModernBertForMaskedLM
from datasets import Dataset, load_dataset
import random

set_seed(42)

# def upsample_texts(texts, factor=1000, shuffle=True, seed=42):
#     out = texts * factor
#     if shuffle:
#         random.Random(seed).shuffle(out)
#     return out

# train_texts = [
#     "Paris is the capital of France.",
#     "The quick brown fox jumps over the lazy dog.",
#     "Transformers are attention-based models.",
#     "Masked language modeling trains bidirectional encoders.",
# ]
# val_texts = [
#     "Berlin is the capital of Germany.",
#     "Attention mechanisms allow tokens to interact.",
# ]

dataset = load_dataset("BioMike/fineweb-text-classification")

texts = list(dataset["train"]["text"])
train_texts = texts[:int(len(texts)*0.9)]
val_texts = texts[int(len(texts)*0.9):]


tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", add_prefix_space=True)

specials = {}
if tokenizer.pad_token_id is None: specials["pad_token"] = "<|pad|>"
if tokenizer.mask_token_id is None: specials["mask_token"] = "<mask>"
if tokenizer.cls_token_id is None: specials["cls_token"] = "<cls>"
if tokenizer.sep_token_id is None: specials["sep_token"] = "<sep>"
if specials:
    tokenizer.add_special_tokens(specials)

cfg = ModernBertConfig(
    vocab_size=len(tokenizer),
    num_hidden_layers=10,
    num_attention_heads=6,
    hidden_size=384,
    intermediate_size=576,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id),
    eos_token_id=(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id),
    cls_token_id=(tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id),
    sep_token_id=(tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id),
)

model = ModernBertForMaskedLM(cfg)
if specials:
    model.resize_token_embeddings(len(tokenizer))

train_ds = Dataset.from_dict({"text": train_texts})
val_ds = Dataset.from_dict({"text": val_texts})

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.3)

args = TrainingArguments(
    output_dir="modern_bert",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    num_train_epochs=10,
    logging_steps=10,
    eval_strategy="epoch",
    save_steps=1000,
    bf16=False,
    save_total_limit=4
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

trainer.train()
trainer.save_model("BiCEbert-mlm-final")
tokenizer.save_pretrained("BiCEbert-mlm-final")
