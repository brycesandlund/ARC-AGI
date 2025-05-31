import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import os
import sys
import multiprocessing

class ArcDataProcessor:
    def __init__(self, max_tokens=10000, max_workers=None):
        self.max_tokens = max_tokens
        
        # Auto-detect maximum workers if not specified
        if max_workers is None:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers
            
        print(f"🔧 Using {self.max_workers} worker threads (CPU cores: {multiprocessing.cpu_count()})")
        
        # Load tokenizer with progress
        print("🔄 Loading tokenizer...")
        with tqdm(total=1, desc="Loading tokenizer", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-0.5B-bnb-4bit")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            pbar.update(1)
        print("✅ Tokenizer loaded successfully!")
        
        # Thread-safe storage
        self.processed_data = []
        self.lock = threading.Lock()
        self.filtered_count = 0
        self.total_processed = 0

    def grid_to_string(self, grid):
        """Convert grid to string representation"""
        rows_str = [str([int(val) for val in row]) for row in grid]
        return "\n".join(rows_str)

    def format_chatml_prompt(self, messages):
        """Format messages into ChatML format"""
        formatted_text = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted_text

    def generate_prompt(self, challenge):
        """Generate user prompt and assistant response from challenge"""
        user_prompt = ("You are a genius at solving IQ tests.\n\n"
                      "Below is a list of input and output pairs with a pattern. "
                      "Your goal is to identify the pattern or transformation in the training examples "
                      "that maps the input to the output, then apply that pattern to the test input "
                      "to give a final output. This is not a maths puzzle, the numbers represent colors. "
                      "This is like a visual IQ test.\n\n"
                      "Respond in the format of the training output examples\n\n"
                      "--Training Examples--")

        # Add training examples (all but the last one)
        for i in range(len(challenge["examples"]) - 1):
            user_prompt += f"\n--Example {i}-- \n\n INPUT: \n\n"
            user_prompt += self.grid_to_string(challenge["examples"][i][0])
            user_prompt += "\n\n\nOUTPUT: \n\n"
            user_prompt += self.grid_to_string(challenge["examples"][i][1])
            user_prompt += "\n\n"

        user_prompt += "\n\n--End of Training Examples--\n\n<test_input>"
        user_prompt += self.grid_to_string(challenge["examples"][-1][0])
        user_prompt += "</test_input>"
        
        # Create the assistant response with the answer
        assistant_response = "<answer>"
        assistant_response += self.grid_to_string(challenge["examples"][-1][1])
        assistant_response += "</answer>"

        return user_prompt, assistant_response

    def process_single_example(self, challenge):
        """Process a single challenge example"""
        try:
            user_prompt, assistant_response = self.generate_prompt(challenge)
            
            # Create messages format
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            
            # Create text format using ChatML
            text_format = self.format_chatml_prompt(messages)
            
            # Tokenize to check length
            tokens = self.tokenizer.encode(text_format, add_special_tokens=True)
            token_count = len(tokens)
            
            with self.lock:
                self.total_processed += 1
                
                # Filter by token length
                if token_count < self.max_tokens:
                    result = {
                        "messages": messages,
                        "text": text_format,
                        "examples": challenge["examples"],
                        "token_count": token_count
                    }
                    self.processed_data.append(result)
                else:
                    self.filtered_count += 1
                    
            return True
            
        except Exception as e:
            print(f"❌ Error processing example: {e}")
            return False

    def process_batch(self, batch_data):
        """Process a batch of examples"""
        results = []
        for challenge in batch_data:
            if self.process_single_example(challenge):
                results.append(True)
            else:
                results.append(False)
        return results

    def process_dataset(self, dataset, target_size=200000):
        """Process the entire dataset with multithreading"""
        print(f"\n🚀 Processing dataset with {self.max_workers} workers...")
        print(f"📊 Target size: {target_size:,} examples")
        print(f"🔢 Max tokens per example: {self.max_tokens:,}")
        
        # Take only the target size from dataset with progress
        dataset_size = min(len(dataset["train"]), target_size)
        print(f"📋 Selecting {dataset_size:,} examples from dataset...")
        
        with tqdm(total=1, desc="Selecting dataset subset", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            dataset_subset = dataset["train"].select(range(dataset_size))
            pbar.update(1)
        
        # Create batches for processing
        batch_size = max(1, len(dataset_subset) // (self.max_workers * 4))  # 4 batches per worker
        batches = []
        
        print(f"📦 Creating batches...")
        with tqdm(total=len(dataset_subset), desc="Creating batches") as pbar:
            for i in range(0, len(dataset_subset), batch_size):
                batch = dataset_subset[i:i + batch_size]
                batches.append(batch)
                pbar.update(len(batch))
        
        print(f"✅ Created {len(batches)} batches of size ~{batch_size}")
        
        # Process batches with ThreadPoolExecutor
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches with enhanced progress bar
            with tqdm(total=len(batches), desc="🔄 Processing batches", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        results = future.result()
                        # Calculate rates
                        elapsed = time.time() - start_time
                        rate = self.total_processed / elapsed if elapsed > 0 else 0
                        keep_rate = (len(self.processed_data) / self.total_processed * 100) if self.total_processed > 0 else 0
                        
                        pbar.set_postfix({
                            'processed': f"{self.total_processed:,}",
                            'kept': f"{len(self.processed_data):,}",
                            'filtered': f"{self.filtered_count:,}",
                            'keep_rate': f"{keep_rate:.1f}%",
                            'rate': f"{rate:.1f}/s"
                        })
                    except Exception as e:
                        print(f"❌ Batch {batch_idx} failed: {e}")
                    finally:
                        pbar.update(1)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📈 Total examples processed: {self.total_processed:,}")
        print(f"✅ Examples kept (< {self.max_tokens:,} tokens): {len(self.processed_data):,}")
        print(f"🚫 Examples filtered out: {self.filtered_count:,}")
        print(f"⚡ Processing rate: {self.total_processed / processing_time:.2f} examples/second")

    def save_datasets(self, output_dir="./arc_processed_data"):
        """Save the processed data in multiple formats"""
        print(f"\n💾 Saving datasets...")
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.processed_data:
            print("❌ No data to save!")
            return
        
        print(f"📊 Saving {len(self.processed_data):,} examples...")
        
        # Save as JSON (conversation format)
        json_path = os.path.join(output_dir, "arc_conversations.json")
        print("💾 Saving conversation format...")
        with tqdm(total=1, desc="Saving JSON conversations", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            with open(json_path, "w") as f:
                json.dump(self.processed_data, f, indent=2)
            pbar.update(1)
        print(f"✅ Saved conversation format to: {json_path}")
        
        # Save as text-only JSON (for text-based training)
        text_data = [{"text": item["text"], "token_count": item["token_count"]} for item in self.processed_data]
        text_path = os.path.join(output_dir, "arc_text_format.json")
        print("💾 Saving text format...")
        with tqdm(total=1, desc="Saving JSON text format", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            with open(text_path, "w") as f:
                json.dump(text_data, f, indent=2)
            pbar.update(1)
        print(f"✅ Saved text format to: {text_path}")
        
        # Create HuggingFace datasets
        print("🤗 Creating HuggingFace datasets...")
        
        # Conversation format dataset
        print("📝 Creating conversation dataset...")
        with tqdm(total=1, desc="Building conversation dataset", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            conversation_dict = {
                "messages": [item["messages"] for item in self.processed_data],
                "examples": [item["examples"] for item in self.processed_data],
                "token_count": [item["token_count"] for item in self.processed_data]
            }
            conversation_dataset = Dataset.from_dict(conversation_dict)
            pbar.update(1)
        
        # Text format dataset
        print("📄 Creating text dataset...")
        with tqdm(total=1, desc="Building text dataset", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            text_dict = {
                "text": [item["text"] for item in self.processed_data],
                "token_count": [item["token_count"] for item in self.processed_data]
            }
            text_dataset = Dataset.from_dict(text_dict)
            pbar.update(1)
        
        # Save locally with progress
        print("💾 Saving HuggingFace datasets locally...")
        with tqdm(total=2, desc="Saving datasets to disk") as pbar:
            conversation_dataset.save_to_disk(os.path.join(output_dir, "conversation_dataset"))
            pbar.update(1)
            pbar.set_description("Saving text dataset")
            text_dataset.save_to_disk(os.path.join(output_dir, "text_dataset"))
            pbar.update(1)
        
        print(f"✅ Saved HuggingFace datasets to: {output_dir}")
        
        # Generate statistics
        print("📊 Generating statistics...")
        self.generate_statistics(output_dir)
        
        return conversation_dataset, text_dataset

    def generate_statistics(self, output_dir):
        """Generate and save statistics about the processed data"""
        if not self.processed_data:
            return
        
        print("📈 Computing statistics...")
        with tqdm(total=1, desc="Computing token statistics", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            token_counts = [item["token_count"] for item in self.processed_data]
            
            stats = {
                "total_examples": len(self.processed_data),
                "token_statistics": {
                    "mean": sum(token_counts) / len(token_counts),
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "median": sorted(token_counts)[len(token_counts) // 2]
                },
                "processing_info": {
                    "max_tokens_filter": self.max_tokens,
                    "total_processed": self.total_processed,
                    "filtered_out": self.filtered_count,
                    "keep_rate": len(self.processed_data) / self.total_processed if self.total_processed > 0 else 0
                }
            }
            pbar.update(1)
        
        stats_path = os.path.join(output_dir, "processing_stats.json")
        with tqdm(total=1, desc="Saving statistics", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            pbar.update(1)
        
        print(f"✅ Saved processing statistics to: {stats_path}")
        print(f"📊 Token statistics: Mean={stats['token_statistics']['mean']:.1f}, "
              f"Min={stats['token_statistics']['min']}, "
              f"Max={stats['token_statistics']['max']}")

def main():
    """Main function to run the data processing"""
    print("🎯 Starting ARC Heavy Dataset Processing...")
    print("=" * 60)
    
    # Load the dataset
    print("📥 Loading dataset...")
    with tqdm(total=1, desc="Loading dataset from HuggingFace", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
        dataset = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")
        pbar.update(1)
    print(f"✅ Dataset loaded. Total examples: {len(dataset['train']):,}")
    
    # Initialize processor with maximum workers
    print("\n🔧 Initializing processor...")
    processor = ArcDataProcessor(max_tokens=10000, max_workers=None)  # None = auto-detect max
    
    # Process the dataset
    print("\n" + "=" * 60)
    processor.process_dataset(dataset, target_size=200000)
    
    # Save the results
    print("\n" + "=" * 60)
    conversation_dataset, text_dataset = processor.save_datasets()
    
    print("\n" + "=" * 60)
    print("🎉 Processing complete!")
    print(f"📊 Final dataset size: {len(conversation_dataset):,} examples")
    
    # Optionally push to HuggingFace Hub
    print("\n" + "=" * 60)
    push_to_hub = input("🤗 Push datasets to HuggingFace Hub? (y/n): ").lower().strip() == 'y'
    
    if push_to_hub:
        hub_name = input("📝 Enter dataset name for HuggingFace Hub: ").strip()
        if hub_name:
            print("🚀 Pushing conversation dataset...")
            with tqdm(total=1, desc="Uploading conversation dataset", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                conversation_dataset.push_to_hub(f"{hub_name}-conversations", private=False)
                pbar.update(1)
            
            print("🚀 Pushing text dataset...")
            with tqdm(total=1, desc="Uploading text dataset", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                text_dataset.push_to_hub(f"{hub_name}-text", private=False)
                pbar.update(1)
            
            print(f"✅ Datasets pushed to HuggingFace Hub as '{hub_name}-conversations' and '{hub_name}-text'")
    
    print("\n🎯 All done! Happy training! 🚀")

if __name__ == "__main__":
    main() 