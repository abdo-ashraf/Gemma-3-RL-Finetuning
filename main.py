import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description="Gemma-3 RL Finetuning")
    parser.add_argument('--config_path', type=str, default="config.json", help='Config file path')
    
    args = parser.parse_args()
    
    train(args.config_path)
    

if __name__ == "__main__":
    main()