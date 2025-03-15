from datasets import load_dataset

# Load and split your dataset
dataset = load_dataset("text", data_files="jarvis_data.txt")

# Save as JSON or Arrow for efficient loading
dataset["train"].to_json("jarvis_data.json")
