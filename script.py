import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

class LlamaFiConfig:
    def __init__(self, model_name, dataset_name, new_model, quantization_config, training_config, peft_config):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model = new_model
        self.quantization_config = quantization_config
        self.training_config = training_config
        self.peft_config = peft_config

class LlamaFi:
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        self.dataset = load_dataset(self.config.dataset_name, split="train")

    def setup_quantization(self):
        self.compute_dtype = getattr(torch, self.config.quantization_config['bnb_4bit_compute_dtype'])
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization_config['use_4bit'],
            bnb_4bit_quant_type=self.config.quantization_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.config.quantization_config['use_nested_quant'],
        )

    def load_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=self.bnb_config,
            device_map={"": 0}
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def setup_training(self):
        self.training_args = TrainingArguments(
            output_dir=self.config.training_config['output_dir'],
            num_train_epochs=self.config.training_config['num_train_epochs'],
            per_device_train_batch_size=self.config.training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config.training_config['gradient_accumulation_steps'],
            optim=self.config.training_config['optim'],
            save_steps=self.config.training_config['save_steps'],
            logging_steps=self.config.training_config['logging_steps'],
            learning_rate=self.config.training_config['learning_rate'],
            weight_decay=self.config.training_config['weight_decay'],
            fp16=self.config.training_config['fp16'],
            bf16=self.config.training_config['bf16'],
            max_grad_norm=self.config.training_config['max_grad_norm'],
            max_steps=self.config.training_config['max_steps'],
            warmup_ratio=self.config.training_config['warmup_ratio'],
            group_by_length=self.config.training_config['group_by_length'],
            lr_scheduler_type=self.config.training_config['lr_scheduler_type'],
            report_to="tensorboard"
        )

    def train(self):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.config.peft_config,
            dataset_text_field="text",
            max_seq_length=self.config.training_config['max_seq_length'],
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=self.config.training_config['packing'],
        )
        self.trainer.train()

    def save(self):
        self.trainer.model.save_pretrained(self.config.new_model)

    def push_to_hub(self, hub_model_name):
        import locale
        locale.getpreferredencoding = lambda: "UTF-8"

        self.model.push_to_hub(hub_model_name, check_pr=True)
        self.tokenizer.push_to_hub(hub_model_name, check_pr=True)


# Define your configuration
config = LlamaFiConfig(
    model_name="NousResearch/Llama-2-7b-chat-hf",
    dataset_name="mlabonne/guanaco-llama2-1k",
    new_model="Llama-2-7b-chat-finetune",
    quantization_config={
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "use_nested_quant": False,
    },
    training_config={
        "output_dir": "./results",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "optim": "paged_adamw_32bit",
        "save_steps": 0,
        "logging_steps": 25,
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "fp16": False,
        "bf16": False,
        "max_grad_norm": 0.3,
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "max_seq_length": None,
        "packing": False,
    },
    peft_config=LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
)

# Initialize LlamaFi
llamfi = LlamaFi(config)

# Run the steps
llamfi.load_dataset()
print("-----completed-----")
llamfi.setup_quantization()
print("-----completed-----")
llamfi.load_model_and_tokenizer()
print("-----completed-----")
llamfi.setup_training()
print("-----completed-----")
llamfi.train()
print("-----completed-----")
llamfi.save()
print("-----completed-----")

# Optionally push to Hugging Face Hub
# llamfi.push_to_hub("your-hub-model-name")
