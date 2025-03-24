# Helper functions to extract answers from different formats
import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# Reward function that checks if the answer is correct
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 50,
        "\n",
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# Reward function that checks if the answer is an integer
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# Reward function that checks if the completion follows the strict format
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward function that checks if the completion follows a more relaxed format
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^{}\n.*?\n{}\n{}\n.*?\n{}\n$".format("<reasoning>", "</reasoning>", "<answer>", "</answer>")
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward function that counts XML tags and penalizes extra content
def count_xml(text) -> float:
    count = 0.0
    if text.count(f"{"<reasoning>"}\n") == 1:
        count += 0.125
    if text.count(f"\n{"</reasoning>"}\n") == 1:
        count += 0.125
    if text.count(f"\n{"<answer>"}\n") == 1:
        count += 0.125
        count -= len(text.split(f"\n{"</answer>"}\n")[-1]) * 0.001
    if text.count(f"\n{"</answer>"}") == 1:
        count += 0.125
        count -= (len(text.split(f"\n{"</answer>"}")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]