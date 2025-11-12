import json
import os
from pathlib import Path

def parse_discord_export(export_folder, output_file='data/training_data.txt'):
    """
    Parse official Discord data export and create training data
    
    Args:
        export_folder: Path to extracted Discord data package
        output_file: Where to save the processed training data
    """
    messages = []
    
    # Discord export has messages in messages/index.json or multiple channel folders
    messages_path = Path(export_folder) / 'messages'
    
    if not messages_path.exists():
        print(f"Error: {messages_path} not found. Make sure you extracted the Discord data package.")
        return
    
    # Look for message files in all channel folders
    for channel_folder in messages_path.iterdir():
        if channel_folder.is_dir():
            messages_file = channel_folder / 'messages.json'
            
            if messages_file.exists():
                print(f"Processing {channel_folder.name}...")
                with open(messages_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # Extract messages
                        for msg in data:
                            content = msg.get('Contents', '').strip()
                            author = msg.get('Author', {}).get('Name', 'Unknown')
                            
                            if content:  # Skip empty messages
                                # Format: "Author: message"
                                messages.append(f"{content}")
                    except json.JSONDecodeError:
                        print(f"Error reading {messages_file}")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(messages))
    
    print(f"\n✅ Processed {len(messages)} messages")
    print(f"✅ Training data saved to {output_file}")
    print(f"✅ File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # Update this path to where you extracted your Discord data
    export_folder = 'data/package'
    
    if not os.path.exists(export_folder):
        print("⚠️  Please extract your Discord data package to 'data/discord_export'")
        print("   Request your data from: User Settings → Privacy & Safety → Request Data")
    else:
        parse_discord_export(export_folder)