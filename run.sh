#!/bin/bash
# Run script for Tamil Political Sentiment Analysis
# Usage: ./run.sh <command> [options]

set -e

# UTF-8 locale for Tamil text display
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Activate virtual environment
source .venv/bin/activate

# Fix SSL certificate issues for HuggingFace downloads
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SSL_CERT_FILE="${SCRIPT_DIR}/micron-ca-bundle.crt"
export REQUESTS_CA_BUNDLE="${SCRIPT_DIR}/micron-ca-bundle.crt"
export CURL_CA_BUNDLE="${SCRIPT_DIR}/micron-ca-bundle.crt"

# HuggingFace token for gated models (DO NOT hardcode)
# Set HF_TOKEN in your environment or in a local .env file (not committed)
export HF_TOKEN="${your_token}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$HF_TOKEN" ]; then
  echo -e "${YELLOW}Warning: HF_TOKEN is not set. Gated Hugging Face models may fail to download.${NC}"
  echo -e "${YELLOW}Set it like: export HF_TOKEN=hf_***  (or put it in .env and source it)${NC}"
fi

print_usage() {
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train-transformer <config>   Train transformer model (muril_base, indicbert, xlm_roberta)"
    echo "  train-llm <config>           Train LLM with LoRA (gemma_9b, qwen_7b)"
    echo "  train-all                    Train all transformer models"
    echo "  predict <model_path>         Generate predictions"
    echo "  ensemble <model1> <model2>   Ensemble predictions from multiple models"
    echo "  analyze-data                 Run data analysis"
    echo "  augment-data                 Augment training data (EDA - simple)"
    echo "  augment-llm --test           Test LLM augmentation (AWS Bedrock)"
    echo "  augment-llm --full           Run full LLM augmentation"
    echo "  analyze-model <model_path>   Generate comprehensive model analysis report"
    echo "  compare-models <m1> <m2>     Compare multiple trained models"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train-transformer muril_base"
    echo "  ./run.sh train-llm gemma_9b"
    echo "  ./run.sh predict outputs/models/muril-base-cased_20240127/best_model"
    echo "  ./run.sh ensemble outputs/models/model1/best_model outputs/models/model2/best_model"
    echo "  ./run.sh analyze-model outputs/models/muril-base-cased_20240127"
    echo "  ./run.sh compare-models outputs/models/model1 outputs/models/model2"
}

train_transformer() {
    CONFIG=$1
    if [ -z "$CONFIG" ]; then
        echo -e "${RED}Error: Please specify a config file${NC}"
        echo "Available configs: muril_base, indicbert, xlm_roberta"
        exit 1
    fi

    CONFIG_PATH="configs/${CONFIG}.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}Training transformer with config: $CONFIG${NC}"

    # Parse config and run
    python -c "
import json
import subprocess
import sys

with open('$CONFIG_PATH') as f:
    config = json.load(f)

cmd = [
    sys.executable, 'src/train_transformer.py',
    '--model_name', config['model_name'],
    '--max_length', str(config['max_length']),
    '--data_dir', config['data_dir'],
    '--output_dir', config['output_dir'],
    '--epochs', str(config['epochs']),
    '--batch_size', str(config['batch_size']),
    '--learning_rate', str(config['learning_rate']),
    '--weight_decay', str(config['weight_decay']),
    '--warmup_ratio', str(config['warmup_ratio']),
    '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
    '--loss_type', config['loss_type'],
    '--focal_gamma', str(config['focal_gamma']),
    '--seed', str(config['seed']),
    '--early_stopping_patience', str(config.get('early_stopping_patience', 5))
]

if config.get('use_augmented', False):
    cmd.append('--use_augmented')

subprocess.run(cmd)
"
}

train_llm() {
    CONFIG=$1
    if [ -z "$CONFIG" ]; then
        echo -e "${RED}Error: Please specify a config file${NC}"
        echo "Available configs: gemma_9b, qwen_7b"
        exit 1
    fi

    CONFIG_PATH="configs/${CONFIG}.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}Training LLM with config: $CONFIG${NC}"

    python -c "
import json
import subprocess
import sys

with open('$CONFIG_PATH') as f:
    config = json.load(f)

cmd = [
    sys.executable, 'src/train_llm.py',
    '--model_name', config['model_name'],
    '--max_length', str(config['max_length']),
    '--data_dir', config['data_dir'],
    '--output_dir', config['output_dir'],
    '--epochs', str(config['epochs']),
    '--batch_size', str(config['batch_size']),
    '--learning_rate', str(config['learning_rate']),
    '--weight_decay', str(config['weight_decay']),
    '--warmup_ratio', str(config['warmup_ratio']),
    '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
    '--eval_epochs', str(config['eval_epochs']),
    '--lora_r', str(config['lora_r']),
    '--lora_alpha', str(config['lora_alpha']),
    '--lora_dropout', str(config['lora_dropout']),
    '--seed', str(config['seed'])
]

if config.get('use_4bit', False):
    cmd.append('--use_4bit')

subprocess.run(cmd)
"
}

train_all() {
    echo -e "${GREEN}Training all transformer models...${NC}"

    for config in muril_base indicbert xlm_roberta; do
        echo -e "${YELLOW}Training $config...${NC}"
        train_transformer $config
        echo -e "${GREEN}Completed $config${NC}"
        echo ""
    done

    echo -e "${GREEN}All models trained!${NC}"
}

predict() {
    MODEL_PATH=$1
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Please specify model path${NC}"
        exit 1
    fi

    echo -e "${GREEN}Generating predictions with model: $MODEL_PATH${NC}"

    python src/inference.py \
        --model_paths "$MODEL_PATH" \
        --model_type transformer \
        --data_dir data \
        --output_dir outputs/predictions \
        --batch_size 32
}

ensemble() {
    if [ "$#" -lt 2 ]; then
        echo -e "${RED}Error: Please specify at least 2 model paths${NC}"
        exit 1
    fi

    echo -e "${GREEN}Ensembling predictions from $# models...${NC}"

    python src/inference.py \
        --model_paths "$@" \
        --model_type transformer \
        --data_dir data \
        --output_dir outputs/predictions \
        --output_name ensemble_submission.csv \
        --ensemble_method soft \
        --batch_size 32
}

analyze_data() {
    echo -e "${GREEN}Running data analysis...${NC}"
    python src/data_utils.py
}

augment_data() {
    echo -e "${GREEN}Running data augmentation (EDA)...${NC}"
    python src/augmentation.py
}

augment_llm() {
    echo -e "${GREEN}Running LLM-based augmentation via AWS Bedrock...${NC}"
    python src/augment_with_llm.py "$@"
}

augment_gemini() {
    echo -e "${GREEN}Running LLM-based augmentation via Vertex AI (Gemini)...${NC}"
    python src/augment_with_gemini.py "$@"
}

analyze_model() {
    MODEL_PATH=$1
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Please specify model path${NC}"
        exit 1
    fi

    echo -e "${GREEN}Generating analysis report for: $MODEL_PATH${NC}"
    python src/analyze.py --model_dir "$MODEL_PATH" --data_dir data
}

compare_models() {
    if [ "$#" -lt 2 ]; then
        echo -e "${RED}Error: Please specify at least 2 model paths${NC}"
        exit 1
    fi

    echo -e "${GREEN}Comparing $# models...${NC}"
    python src/analyze.py --compare "$@" --output_dir outputs/analysis
}

# Main command router
case "$1" in
    train-transformer)
        train_transformer "$2"
        ;;
    train-llm)
        train_llm "$2"
        ;;
    train-all)
        train_all
        ;;
    predict)
        predict "$2"
        ;;
    ensemble)
        shift
        ensemble "$@"
        ;;
    analyze-data)
        analyze_data
        ;;
    augment-data)
        augment_data
        ;;
    augment-llm)
        shift
        augment_llm "$@"
        ;;
    augment-gemini)
        shift
        augment_gemini "$@"
        ;;
    analyze-model)
        analyze_model "$2"
        ;;
    compare-models)
        shift
        compare_models "$@"
        ;;
    *)
        print_usage
        ;;
esac
