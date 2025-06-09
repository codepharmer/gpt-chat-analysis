import json
import random

# Read the JSON file
with open('chatgpt_062025/conversations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Make sure data is a list/array
if not isinstance(data, list):
    raise ValueError("The JSON file must contain an array as its main object")

# Sample 20 items randomly
# If there are fewer than 20 items, take all of them
sample_size = min(3, len(data))
sampled_data = random.sample(data, sample_size)

# Save the sampled data to a new file
with open('sampled_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"Successfully sampled {sample_size} items and saved to sampled_conversations.json")
