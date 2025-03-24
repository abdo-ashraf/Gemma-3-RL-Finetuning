from datasets import load_dataset, Dataset

# For GSM8K dataset, We notice all answers like about have a ####, so we extract it
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Function to prepare the GSM8K dataset
def get_gsm8k_questions(system_prompt:str) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return dataset