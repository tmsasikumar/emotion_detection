import os

def create_directories():
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../data/features', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
