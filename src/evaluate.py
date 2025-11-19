
import mlflow
# Import required components for inference
import torch
from typing import Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from typing import Dict, Optional, Tuple, Any
from trl import TrlParser
from datasets import load_from_disk,Dataset
from data import process_xlam_sample
import multiprocessing
from mlflow.genai.scorers import Correctness, Equivalence
from utils import set_custom_env

@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model path to load the model from"}
    )
    

    eval_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval dataset"}
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



def load_trained_model(model_path:str,
                       compute_dtype: torch.dtype,
                       attn_implementation: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a trained model with LoRA adapters for inference.
    
    This function loads the base model with quantization and applies the trained
    LoRA adapters for efficient inference. It's designed to work after training
    completion or for loading previously saved models.
    
    Args:
        model_config (ModelConfig): Configuration for the base model
        adapter_path (str): Path to the saved LoRA adapter
        compute_dtype (torch.dtype): Computation data type
        attn_implementation (str): Attention implementation
        
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
        
    Note:
        You may need to restart the notebook to free GPU memory before loading
        the model for inference, especially after training.
    """
    print(f"🔄 Loading trained model")
    
    
    # Load tokenizer with proper configuration
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True
        )
    print(f"🔤 Tokenizer loaded")
    
    # Load base model
    print(f"📦 Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    
    # Enable evaluation mode
    model.eval()
    
    print("✅ Model loaded successfully and ready for inference!")
    print(f"💾 Total memory usage: ~{model.get_memory_footprint() / 1e9:.1f} GB")
    
    return model, tokenizer

def generate_function_call(model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer, 
                          input: str,
                          max_new_tokens: int = 512,
                          temperature: float = 0.7,
                          do_sample: bool = True) -> str:
    """
    Generate a function call response using the fine-tuned model.
    
    Args:
        model (AutoModelForCausalLM): Fine-tuned model with LoRA adapters
        tokenizer (AutoTokenizer): Model tokenizer
        input (str): Input text for function calling
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature (only used when do_sample=True)
        do_sample (bool): Whether to use sampling
        
    Returns:
        str: Generated response with function calls
        
    Example Prompt Format:
        "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
    """
    prompt = f"""<user>{input}</user>\n\n<tools>"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response with proper parameter handling
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    
    # Only add sampling parameters if do_sample=True
    if do_sample:
        generation_kwargs["temperature"] = temperature
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # Decode result
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_ds(dataset:Dataset,tokenizer:AutoTokenizer,model:AutoModelForCausalLM)->Dataset:

    def process_batch(batch):
        """Process a batch of samples with the tokenizer."""
        processed_batch = []
        for i in range(len(batch['query'])):
            row = {
                    'query': batch['query'][i],
                    'tools': batch['tools'][i],
                    'answers': batch['answers'][i]
                }
            processed_row = process_xlam_sample(row, tokenizer)
            processed_row['generated'] = generate_function_call(model, tokenizer, processed_row['query'])

            processed_batch.append(processed_row)

            # Convert to batch format
            return {
                'inputs': [item['query'] for item in processed_batch],
                'expectations': [item['text'] for item in processed_batch],
                'outputs': [item['generated'] for item in processed_batch]
            }

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # Process in batches for efficiency
        num_proc=min(4, multiprocessing.cpu_count()),  # Use multiple cores
        desc="Processing xLAM samples"
    )
    return dataset

if __name__ == "__main__":
    # Load the model
    parser = TrlParser((ScriptArguments))
    script_args = parser.parse_args_and_config()

    custom_env: Dict[str, str] = {
                                   "MLFLOW_TRACKING_URI": script_args.mlflow_tracking_server,
                                   "MLFLOW_EXPERIMENT_NAME":script_args.mlflow_exp_name,
                                   "MLFLOW_RUN_ID": script_args.mlflow_run_id                              
                                }
    set_custom_env(custom_env)

    mlflow.set_tracking_uri(script_args.mlflow_tracking_server)
    mlflow.enable_system_metrics_logging()

    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'

    trained_model, trained_tokenizer = load_trained_model(
        model_path=script_args.model_path,
        compute_dtype=compute_dtype,
        attn_implementation=attn_implementation
    )

    eval_ds = load_from_disk(script_args.train_path)
    eval_dataset_processed = process_ds(eval_ds, trained_tokenizer, trained_model)

    results = mlflow.genai.evaluate(
    data=eval_dataset_processed,
    scorers=[
        Correctness(),
        Equivalence()
    ]
    )




