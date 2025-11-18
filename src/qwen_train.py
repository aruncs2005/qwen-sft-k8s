from dataclasses import dataclass, field
from typing import Optional

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments
from trl import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Tuple, Any

from trl import SFTTrainer

import mlflow
from utils import set_custom_env
import sys
from data import process_dataset, preview_dataset_sample

@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    train_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train dataset"}
    )

    eval_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval dataset"}
    )

    model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID to use for SFT training"}
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

    
    mlflow_tracking_server: str = field(
        default="",
        metadata={"help": "SageMaker managed MLflow tracking server"}
    )

    mlflow_exp_name: str = field(
        default="",
        metadata={"help": "SageMaker managed MLflow experiment name"})

    mlflow_run_id: str = field(
        default="",
        metadata={"help": "SageMaker managed MLflow run id"})


def setup_tokenizer(script_args:ScriptArguments):
    tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_id,
            use_fast=True
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_dataset(script_args:ScriptArguments,tokenizer)->Tuple:
    
    train_ds = load_from_disk(script_args.train_path)
    eval_ds = load_from_disk(script_args.eval_path)

    processed_train_ds = process_dataset(tokenizer,train_ds)
    processed_eval_ds = process_dataset(tokenizer,eval_ds)

    print("***** Sample data from Training set*****")
    preview_dataset_sample(processed_train_ds, 0)
    print("***** Sample data from Eval set*****")
    preview_dataset_sample(processed_eval_ds,0)

    return processed_train_ds,processed_eval_ds


def load_model(script_args:ScriptArguments)-> Any:

    model = AutoModelForCausalLM.from_pretrained(
    script_args.model_id,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
    trust_remote_code=True             
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )

    model = get_peft_model(model, peft_config)
    return model



def training_function(script_args, training_args):
    ################
    # Dataset
    ################
    
    # Load datasets
    tokenizer = setup_tokenizer(script_args)

    train_dataset,eval_dataset = create_dataset(script_args,tokenizer)

    #Load Peft model
    model = load_model(script_args)
    training_args.dataset_text_field="text"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset

    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(training_args.output_dir,"qwen3"))
    tokenizer.save_pretrained(os.path.join(training_args.output_dir,"qwen3"))
    print("*** Model Saved")
    
    

if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    #Enable mlflow tracking
    mlflow.set_tracking_uri(script_args.mlflow_tracking_server)
    mlflow.enable_system_metrics_logging()
        
    custom_env: Dict[str, str] = {"HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
                                   "MLFLOW_TRACKING_URI": script_args.mlflow_tracking_server,
                                   "MLFLOW_EXPERIMENT_NAME":script_args.mlflow_exp_name,
                                   "MLFLOW_RUN_ID": script_args.mlflow_run_id,
                                   "CUDA_VISIBLE_DEVICES": str(torch.cuda.device_count())
                                }
    set_custom_env(custom_env)

    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)
    torch.distributed.destroy_process_group()
    sys.exit(0)