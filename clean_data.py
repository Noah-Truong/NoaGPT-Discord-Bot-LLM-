import re
from config import Config

def clean_discord_data(input_file, output_file, min_length=15):
    """
    Clean Discord chat data for LLM training
    
    Args:
        input_file: Path to raw Discord data
        output_file: Path to save cleaned data
        min_length: Minimum character length for a valid line
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Step 1: Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Step 2: Remove Discord mentions (@user, <@123456>)
    text = re.sub(r'<@!?\d+>', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Step 3: Remove Discord channel references (#channel, <#123456>)
    text = re.sub(r'<#\d+>', '', text)
    
    # Step 4: Remove Discord emoji codes (<:emoji:123456>)
    text = re.sub(r'<a?:\w+:\d+>', '', text)
    
    # Step 5: Remove standalone emojis and emoji spam
    # This removes lines that are mostly emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Step 6: Fix common encoding issues
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '-',
        'â€"': '--',
        'â€¦': '...',
        'Â': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Step 7: Remove bot commands (starts with !, /, ?, -)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip if line starts with bot command
        if line.startswith(('!', '/', '?', '-', '.', '>', '<')):
            continue
        
        # Skip if line is too short
        if len(line) < min_length:
            continue
        
        # Skip if line is mostly emojis
        if emoji_pattern.sub('', line).strip() == '':
            continue
        
        # Skip lines with excessive special characters
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / max(len(line), 1)
        if special_char_ratio > 0.5:
            continue
        
        # Remove excessive whitespace
        line = ' '.join(line.split())
        
        # Skip if it's just repeated characters
        if len(set(line.replace(' ', ''))) < 3:
            continue
        
        # Add capitalization if missing (helps model learn proper grammar)
        if line and line[0].islower():
            line = line[0].upper() + line[1:]
        
        # Add period if missing (helps model learn sentence structure)
        if line and line[-1] not in '.!?':
            line = line + '.'
        
        cleaned_lines.append(line)
    
    # Step 8: Group short lines into paragraphs (better context)
    paragraphs = []
    current_paragraph = []
    current_length = 0
    
    for line in cleaned_lines:
        current_paragraph.append(line)
        current_length += len(line)
        
        # Create paragraph when we have ~200 characters or hit natural break
        if current_length > 200 or any(word in line.lower() for word in ['however', 'but', 'also', 'because']):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
            current_length = 0
    
    # Add remaining lines
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Step 9: Remove duplicate paragraphs
    paragraphs = list(dict.fromkeys(paragraphs))
    
    # Step 10: Filter out very repetitive text
    final_paragraphs = []
    for para in paragraphs:
        words = para.split()
        if len(words) < 5:
            continue
        
        # Check for excessive repetition
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.3:  # At least 30% unique words
            final_paragraphs.append(para)
    
    # Combine paragraphs with double newlines
    cleaned_text = '\n\n'.join(final_paragraphs)
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Print statistics
    print("=" * 50)
    print("Data Cleaning Complete!")
    print("=" * 50)
    print(f"Original lines: {len(lines)}")
    print(f"Cleaned lines: {len(cleaned_lines)}")
    print(f"Final paragraphs: {len(final_paragraphs)}")
    print(f"Original size: {len(text) / 1024:.2f} KB")
    print(f"Cleaned size: {len(cleaned_text) / 1024:.2f} KB")
    print(f"Reduction: {(1 - len(cleaned_text) / len(text)) * 100:.1f}%")
    print("=" * 50)
    
    return cleaned_text

def preview_cleaned_data(file_path, num_samples=5):
    """Preview some samples from cleaned data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n\n')
    
    print("\n" + "=" * 50)
    print(f"Preview of Cleaned Data ({num_samples} samples):")
    print("=" * 50 + "\n")
    
    for i, para in enumerate(paragraphs[:num_samples], 1):
        print(f"Sample {i}:")
        print(para)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    # Clean the data
    input_file = Config.RAW_DATA
    output_file = 'data/training_data_cleaned.txt'
    
    print("Starting data cleaning...")
    cleaned_text = clean_discord_data(input_file, output_file, min_length=15)
    
    # Preview results
    preview_cleaned_data(output_file)
    
    print("\nCleaned data saved to:", output_file)
    print("You can now use this file for training!")
    print("\nTo use it in your training script, update this line:")
    print("with open('data/training_data_cleaned.txt', 'r') as f:")
    print("    training_text = f.read()")