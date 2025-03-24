import json
from unsloth import FastLanguageModel
from utils.load_gsm8k_dataset import get_gsm8k_questions
from utils.reward_funcs import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func
from trl import GRPOConfig, GRPOTrainer

# Define the system prompt that instructs the model to use a specific format
reasoning_start = "<reasoning>"
reasoning_end   = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

def train(config_path="./config.json"):
    # Load the config file
    with open(config_path, "r") as file:
        config = json.load(file)

    dataset = get_gsm8k_questions(system_prompt=SYSTEM_PROMPT)

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model_id'],
    max_seq_length=config['max_seq_length'],
    load_in_4bit=True,  # Enables memory-efficient training
    fast_inference=True,  # Enable vLLM fast inference
)
    
    ## Setup LoRA model
    model = FastLanguageModel.get_peft_model(
    model,
    r=config['lora_r'],  # Adjustable, 16 is a good starting point
    lora_alpha=config['lora_r'],  # Should be at least r
    lora_dropout=config['lora_dropout'],
    target_modules=config['target_lora_modules'],
    use_gradient_checkpointing="unsloth",  # Memory-efficient fine-tuning
    bias="none",  # No additional trainable bias
    random_state=config['seed'],
    )

    print(model.print_trainable_parameters())

    training_args = GRPOConfig(
    learning_rate=config['learning_rate'],
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=config['logging_steps'],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=config['max_prompt_length'],
    max_completion_length=config['max_seq_length'] - config['max_prompt_length'],
    max_steps=config['max_steps'],
    save_steps=config['save_steps'],
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir=config["outputs"],
    fp16=True,
    # bf16=True,
    # use_vllm=True,
    # vllm_dtype='float16',
    # vllm_device='cuda:1'
)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train model
    trainer.train()