import json
import multiprocessing
from datasets import load_dataset, Dataset
from typing import Dict, Any,Optional

from transformers import AutoTokenizer

def process_xlam_sample(row: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Process a single xLAM dataset sample into training format.
    
    The format we create is:
    <user>[user query]</user>
    
    <tools>
    [tool definitions]
    </tools>
    
    <calls>
    [expected function calls]
    </calls>[EOS_TOKEN]
    """
    # Format user query
    formatted_query = f"<user>{row['query']}</user>\n\n"

    # Parse and format available tools
    try:
        parsed_tools = json.loads(row["tools"])
        tools_text = '\n'.join(str(tool) for tool in parsed_tools)
    except json.JSONDecodeError:
        tools_text = str(row["tools"])  # Fallback to raw string
    
    formatted_tools = f"<tools>{tools_text}</tools>\n\n"

    # Parse and format expected function calls
    try:
        parsed_answers = json.loads(row["answers"])
        answers_text = '\n'.join(str(answer) for answer in parsed_answers)
    except json.JSONDecodeError:
        answers_text = str(row["answers"])  # Fallback to raw string

    formatted_answers = f"<calls>{answers_text}</calls>"

    # Combine all parts with EOS token
    complete_text = formatted_query + formatted_tools + formatted_answers + tokenizer.eos_token

    # Update row with processed data
    row["query"] = formatted_query
    row["tools"] = formatted_tools
    row["answers"] = formatted_answers
    row["text"] = complete_text

    return row


def process_dataset(tokenizer: AutoTokenizer, dataset: Dataset, sample_size: Optional[int] = None) -> Dataset:
    """
    Process the dataset using multiprocessing for efficiency.

    Args:
        tokenizer: Configured tokenizer for the model
        dataset: The dataset to process
        sample_size: Optional number of samples to use (None for full dataset)

    Returns:
        Dataset: Processed dataset ready for training
    """
    # Sample dataset if requested (useful for testing)
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))

    # Process all samples using multiprocessing for efficiency
    print("⚙️ Processing dataset samples into training format...")

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
            processed_batch.append(processed_row)

        # Convert to batch format
        return {
            'text': [item['text'] for item in processed_batch],
            'query': [item['query'] for item in processed_batch],
            'tools': [item['tools'] for item in processed_batch],
            'answers': [item['answers'] for item in processed_batch]
        }

    # Process the dataset
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=min(4, multiprocessing.cpu_count()),  # Use multiple cores
        desc="Processing xLAM samples"
    )

    print("✅ Dataset processing complete!")
    print(f"📊 Final dataset size: {len(processed_dataset):, } samples")
    print(f"🔤 Average text length: {sum(len(text) for text in processed_dataset['text']) / len(processed_dataset):.2f}")

    return processed_dataset



def preview_dataset_sample(dataset: Dataset, index: int = 0) -> None:
    """
    Display a formatted preview of a dataset sample for inspection.
    
    Args:
        dataset: The processed dataset
        index: Index of the sample to preview (default: 0)
    """
    if index >= len(dataset):
        print(f"❌ Index {index} is out of range. Dataset has {len(dataset)} samples.")
        return
    
    sample = dataset[index]
    
    print(f"📋 Dataset Sample Preview (Index: {index})")
    print("=" * 80)
    
    print(f"\n🔍 Raw Components:")
    print(f"Query: {sample['query'][:200]}{'...' if len(sample['query']) > 200 else ''}")
    print(f"Tools: {sample['tools'][:200]}{'...' if len(sample['tools']) > 200 else ''}")
    print(f"Answers: {sample['answers'][:200]}{'...' if len(sample['answers']) > 200 else ''}")
    
    print(f"\n📝 Complete Training Text:")
    print("-" * 40)
    print(sample['text'])
    print("-" * 40)
    
    print(f"\n📊 Sample Statistics:")
    print(f"   • Text length: {len(sample['text']):,} characters")
    print(f"   • Estimated tokens: ~{len(sample['text']) // 4:,} tokens")
    
    print("\n✅ Preview complete!")