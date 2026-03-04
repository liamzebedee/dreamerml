#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r "$(dirname "$0")/requirements.txt"

echo "Downloading TinyStories-33M model..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')
AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
print('Model cached successfully.')
"

echo "Setup complete."
