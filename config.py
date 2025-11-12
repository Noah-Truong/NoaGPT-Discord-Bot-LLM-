class Config:
    # Model parameters
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    MAX_SEQ_LEN = 512
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    
    # Data parameters
    MAX_LENGTH = 128
    
    # Paths
    MODEL_SAVE_PATH = 'checkpoints/online_learning.pth'
    TOKENIZER_SAVE_PATH = 'checkpoints/tokenizer.pkl'
    RAW_DATA = 'data/training_data.txt'
    MOTHERLOAD_DATA_PATH = 'mixed_data/motherload_data.txt'
    MIXED_DATA_FOLDER = 'mixed_data'