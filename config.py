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
    MODEL_SAVE_PATH = 'checkpoints/best_model.pth'
    TOKENIZER_SAVE_PATH = 'checkpoints/tokenizer.pkl'
    DATA_PATH = 'data/training_data_mixed.txt'
