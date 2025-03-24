import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description="Gemma-3 RL Finetuning")
    parser.add_argument('--config_path', type=str, help='Learning rate for the optimizer')
    
    args = parser.parse_args()
    
    train(args.config_path)
    

if __name__ == "__main__":
    main()