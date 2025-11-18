# Workflow
# Create prompt template(s)
# Define and register your prompt templates in MLflow Prompt Registry for version control and easy access.

# Prepare evaluation dataset
# Create test cases with inputs and expected outcomes to systematically evaluate prompt performance.

# Define a wrapper function to generate responses
# Wrap your prompt in a function that takes dataset inputs and generates responses using your model.

# Define evaluation scorers
# Set up built-in and custom scorers to measure quality, accuracy, and task-specific criteria.

# Run evaluation
# Execute the evaluation and review results in MLflow UI to analyze performance and iterate.


import mlflow

# Define prompt templates. MLflow supports both text and chat format prompt templates.
PROMPT_V1 = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer the following question.",
    },
    {
        "role": "user",
        # Use double curly braces to indicate variables.
        "content": "Question: {{question}}",
    },
]

# Register the prompt template to the MLflow Prompt Registry for version control
# and convenience of loading the prompt template. This is optional.
mlflow.genai.register_prompt(
    name="qa_prompt",
    template=PROMPT_V1,
    commit_message="Initial prompt",
)


from openai import OpenAI

client = OpenAI()

def load_model(local_path:str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return model, tokenizer


@mlflow.trace
def predict_fn(question: str) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/qa_prompt@latest")
    rendered_prompt = prompt.format(question=question)

    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=rendered_prompt
    )
    return response.choices[0].message.content