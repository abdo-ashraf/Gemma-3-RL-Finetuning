# Gemma-3-RL-Finetuning

## Overview
This project fine-tunes the [Gemma-3-1B-IT](https://ai.google.dev/gemma) model using Reinforcement Learning (RL) with the **Group Relative Policy Optimization (GRPO)** algorithm. The goal is to enhance the model's reasoning and solution generation for mathematical problem-solving using the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k).

## Features
- **LoRA fine-tuning** with [Unsloth](https://github.com/unslothai/unsloth) for efficient training.
- **GRPO reward-based training** with multiple reward functions.
- **Dataset processing** for GSM8K, extracting structured answers.
- **Custom reward functions** to ensure correct formatting and correctness.
- **Fast inference support** with vLLM.

## Project Structure
```
└── abdo-ashraf-gemma-3-rl-finetuning/
    ├── README.md            # Project documentation
    ├── LICENSE              # MIT License
    ├── __init__.py          # Package initialization
    ├── config.json          # Training configurations
    ├── main.py              # Entry point to start training
    ├── requirements.txt     # Dependencies
    ├── train.py             # Training script
    ├── Notebooks/           # Jupyter notebooks for analysis
    ├── outputs/             # Model checkpoints and logs
    └── utils/               # Utility scripts
        ├── __init__.py      
        ├── load_gsm8k_dataset.py  # Dataset loader
        └── reward_funcs.py   # Reward functions for RL
```

## Installation
### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/abdo-ashraf-gemma-3-rl-finetuning.git
   cd abdo-ashraf-gemma-3-rl-finetuning
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running Training
To start the training process, run:
```bash
python main.py --config_path config.json
```

### Configuration
Modify `config.json` to adjust parameters such as:
- `model_id`: Pretrained model ID
- `max_seq_length`: Sequence length for training
- `lora_r`: Rank for LoRA fine-tuning
- `learning_rate`: Learning rate for GRPO training
- `max_steps`: Total training steps

## Reward Functions
Reward functions guide reinforcement learning by scoring model outputs:
- **correctness_reward_func**: Rewards correct answers.
- **int_reward_func**: Rewards integer answers.
- **strict_format_reward_func**: Ensures strict XML formatting.
- **soft_format_reward_func**: Allows flexible formatting.
- **xmlcount_reward_func**: Penalizes extra content outside XML tags.

## Model Checkpoints and Outputs
- Trained models and logs are saved in the `outputs/` directory.
- Modify `output_dir` in `config.json` to change the save location.

## Fine-Tuning Results
For same question and system prompt

**System Prompt:**
```
You are given a problem.
Think about the problem and provide your working out.
Place it between <reasoning> and </reasoning>.
Then, provide your solution between <answer></answer>
```
**Question:**
```
Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
```

### Before Fine-Tuning
**Model Response:**
```
<answer>100.0</answer>
```

---

### After Fine-Tuning
**Model Response:**
```
<reasoning>
Let the cost of the wallet be $100.
Betty has half of the money she needs, so she has $\frac{1}{2} \times 100 = 50$.
Her parents gave her $15.
Her grandparents gave her twice as much as her parents, so grandparents gave her $2 \times 15 = 30$.
The total amount of money her grandparents gave her is $30$.
So, the total amount of money Betty has is $50 + 15 + 30 = 95$.
The amount of money she needs is $100 - 95 = 5$.
Therefore, Betty needs $5 more to buy the wallet.
</reasoning>
<answer>5</answer>
```

---

This demonstrates that Fine-tuning significantly improved the model’s ability to break down the problem and provide a structured response with correct reasoning.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- [Hugging Face R1 Open course](https://huggingface.co/learn/nlp-course/en/chapter12/1?fw=pt)
- [Google Gemma](https://ai.google.dev/gemma)
- [Unsloth](https://github.com/unslothai/unsloth)
- [TRL by Hugging Face](https://huggingface.co/docs/trl)
- [OpenAI GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)

## Contact
For questions, feel free to reach out:
- **Author**: Abdelrhman Ashraf
- **Email**: abdoashraff185@gmail.com

