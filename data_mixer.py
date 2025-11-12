import requests
import random
import os

def download_quality_text():
    """
    Download high-quality text from various free sources
    Returns a dictionary of text by category
    """
    texts = {}
    
    print("Downloading quality text samples...")
    
    # 1. Project Gutenberg - Classic Literature
    gutenberg_books = {
        'pride_and_prejudice': 'https://www.gutenberg.org/files/1342/1342-0.txt',
        'alice_in_wonderland': 'https://www.gutenberg.org/files/11/11-0.txt',
        'sherlock_holmes': 'https://www.gutenberg.org/files/1661/1661-0.txt',
    }
    
    for name, url in gutenberg_books.items():
        try:
            print(f"  Downloading {name}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = response.text
                # Remove Gutenberg header/footer
                start = text.find('***')
                end = text.rfind('***')
                if start != -1 and end != -1:
                    text = text[start:end]
                texts[name] = text[:50000]  # First 50k characters
        except Exception as e:
            print(f"    Failed to download {name}: {e}")
    
    # 2. Sample news/articles style text
    sample_articles = """
    Technology continues to advance at a rapid pace, transforming how we live and work. Artificial intelligence 
    has become increasingly sophisticated, enabling machines to perform tasks that once required human intelligence. 
    From natural language processing to computer vision, AI systems are becoming more capable each year.
    
    The importance of education in modern society cannot be overstated. As technology evolves, the skills required 
    in the workforce continue to change. Lifelong learning has become essential for professional success. Many 
    professionals now engage in continuous education through online courses, workshops, and self-directed study.
    
    Environmental conservation efforts are gaining momentum worldwide. Climate change has become a pressing concern 
    for governments and citizens alike. Renewable energy sources such as solar and wind power are becoming more 
    efficient and cost-effective. Many countries are setting ambitious goals to reduce carbon emissions.
    
    Social media has fundamentally changed how people communicate and share information. While these platforms 
    enable instant connection across vast distances, they also present challenges related to privacy, misinformation, 
    and mental health. Understanding the impact of social media on society remains an important area of research.
    
    Space exploration continues to capture human imagination. Recent missions to Mars have provided valuable data 
    about the red planet's geology and potential for supporting life. Private companies are now working alongside 
    government agencies to advance space technology and make space travel more accessible.
    """
    
    texts['articles'] = sample_articles
    
    # 3. Sample conversational but well-structured text
    sample_conversations = """
    "How was your day today?" she asked, genuinely interested in hearing about his experiences.
    
    "It was quite productive, actually," he replied. "I managed to finish the project I've been working on for weeks. 
    The sense of accomplishment was incredible. How about you? Did you make progress on your goals?"
    
    She nodded enthusiastically. "I did! I finally completed that online course I started last month. The material 
    was challenging, but I learned so much about data analysis and visualization."
    
    "That's fantastic," he said with a smile. "I've been thinking about taking a similar course myself. Would you 
    recommend it for someone without much technical background?"
    
    "Absolutely," she assured him. "The instructors do an excellent job of explaining complex concepts in accessible 
    ways. They start with the fundamentals and gradually build up to more advanced topics."
    
    They continued discussing their learning experiences, sharing insights and recommendations. The conversation 
    flowed naturally, covering topics ranging from professional development to personal interests.
    """
    
    texts['conversations'] = sample_conversations
    
    return texts

def mix_training_data(discord_file, output_file, discord_ratio=0.3):
    """
    Mix cleaned Discord data with quality text
    
    Args:
        discord_file: Path to cleaned Discord data
        output_file: Path to save mixed data
        discord_ratio: Proportion of Discord data (0.0 to 1.0)
                      0.3 = 30% Discord, 70% quality text
    """
    
    print("\n" + "=" * 50)
    print("Starting Data Mixing Process")
    print("=" * 50 + "\n")
    
    # Load cleaned Discord data
    print("Loading cleaned Discord data...")
    try:
        with open(discord_file, 'r', encoding='utf-8') as f:
            discord_text = f.read()
        print(f"  Loaded {len(discord_text) / 1024:.2f} KB of Discord data")
    except FileNotFoundError:
        print(f"Error: {discord_file} not found!")
        print("Please run the cleaning script first.")
        return
    
    # Download quality text
    quality_texts = download_quality_text()
    
    if not quality_texts:
        print("\nWarning: Could not download quality text. Using only Discord data.")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(discord_text)
        return
    
    # Combine quality texts
    print("\nCombining quality text sources...")
    quality_text = '\n\n'.join(quality_texts.values())
    print(f"  Combined {len(quality_text) / 1024:.2f} KB of quality text")
    
    # Calculate target sizes
    discord_paragraphs = [p for p in discord_text.split('\n\n') if p.strip()]
    quality_paragraphs = [p for p in quality_text.split('\n\n') if p.strip()]
    
    # Calculate how many paragraphs of each type we need
    total_discord = len(discord_paragraphs)
    total_quality = len(quality_paragraphs)
    
    # Adjust ratio based on available data
    if total_discord > 0 and total_quality > 0:
        discord_sample_size = int(total_discord * discord_ratio / (1 - discord_ratio))
        quality_sample_size = total_quality
        
        # If we don't have enough quality text, adjust
        if discord_sample_size > total_discord:
            discord_sample_size = total_discord
            quality_sample_size = int(total_discord * (1 - discord_ratio) / discord_ratio)
        
        # Sample paragraphs
        discord_sample = random.sample(discord_paragraphs, min(discord_sample_size, total_discord))
        quality_sample = random.sample(quality_paragraphs, min(quality_sample_size, total_quality))
        
        # Combine and shuffle
        all_paragraphs = discord_sample + quality_sample
        random.shuffle(all_paragraphs)
        
        # Create final text
        mixed_text = '\n\n'.join(all_paragraphs)
        
        # Save mixed data
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mixed_text)
        
        # Calculate actual ratio
        discord_chars = sum(len(p) for p in discord_sample)
        quality_chars = sum(len(p) for p in quality_sample)
        total_chars = discord_chars + quality_chars
        actual_ratio = discord_chars / total_chars if total_chars > 0 else 0
        
        # Print statistics
        print("\n" + "=" * 50)
        print("Data Mixing Complete!")
        print("=" * 50)
        print(f"Discord paragraphs: {len(discord_sample)}")
        print(f"Quality paragraphs: {len(quality_sample)}")
        print(f"Total paragraphs: {len(all_paragraphs)}")
        print(f"\nDiscord content: {actual_ratio * 100:.1f}%")
        print(f"Quality content: {(1 - actual_ratio) * 100:.1f}%")
        print(f"\nFinal size: {len(mixed_text) / 1024:.2f} KB")
        print(f"Saved to: {output_file}")
        print("=" * 50)
        
    else:
        print("Error: Not enough data to mix!")

def preview_mixed_data(file_path, num_samples=3):
    """Preview samples from mixed data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n\n')
    
    print("\n" + "=" * 50)
    print(f"Preview of Mixed Training Data ({num_samples} samples):")
    print("=" * 50 + "\n")
    
    # Show random samples
    samples = random.sample(paragraphs, min(num_samples, len(paragraphs)))
    
    for i, para in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(para[:200] + "..." if len(para) > 200 else para)
        print("-" * 50 + "\n")

def create_fallback_quality_text(output_file):
    """
    Create quality text from built-in samples if download fails
    """
    print("Creating fallback quality text from built-in samples...")
    
    fallback_text = """
    The pursuit of knowledge has been a fundamental human endeavor throughout history. From ancient philosophers 
    to modern scientists, people have sought to understand the world around them. This curiosity drives innovation 
    and progress in every field of human achievement.
    
    Effective communication is essential in both personal and professional contexts. The ability to express ideas 
    clearly and listen actively can significantly impact relationships and career success. Many successful leaders 
    attribute their achievements to strong communication skills developed over years of practice.
    
    Technology has revolutionized how we access information and connect with others. The internet has made vast 
    amounts of knowledge available at our fingertips. However, this abundance of information also requires critical 
    thinking skills to evaluate sources and distinguish fact from fiction.
    
    Physical and mental health are interconnected aspects of overall wellbeing. Regular exercise, adequate sleep, 
    and proper nutrition contribute to both physical fitness and mental clarity. Many people find that maintaining 
    healthy habits improves their productivity and quality of life.
    
    The arts play a vital role in society, providing entertainment, inspiration, and cultural expression. Whether 
    through music, literature, visual arts, or performance, creative works enrich human experience and foster 
    emotional connection. Supporting the arts benefits communities in countless ways.
    
    Environmental sustainability has become increasingly important as we face global challenges. Conservation efforts, 
    renewable energy adoption, and sustainable practices can help protect natural resources for future generations. 
    Individual actions, when combined with systemic changes, can make a significant difference.
    
    Financial literacy is a crucial life skill that everyone should develop. Understanding concepts like budgeting, 
    saving, investing, and debt management enables people to make informed financial decisions. Many schools now 
    recognize the importance of including financial education in their curricula.
    
    Problem-solving abilities are valuable in virtually every aspect of life. Breaking down complex challenges into 
    manageable steps, considering multiple perspectives, and testing potential solutions are skills that can be 
    developed and refined through practice and experience.
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fallback_text)
    
    return fallback_text

if __name__ == "__main__":
    # Configuration
    discord_file = 'data/training_data_cleaned.txt'
    mixed_data_index = 0
    output_file = 'mixed_data/training_data_mixed.txt'
    while os.path.exists(output_file):
        mixed_data_index += 1
        output_file = f'mixed_data/training_data_mixed_MK{mixed_data_index}.txt'

    discord_ratio = 0.3  # 30% Discord, 70% quality text
    
    print("Discord Data Mixer")
    print("This will create a balanced training dataset")
    print(f"Target ratio: {discord_ratio * 100:.0f}% Discord, {(1 - discord_ratio) * 100:.0f}% quality text\n")
    
    # Check if cleaned Discord data exists
    if not os.path.exists(discord_file):
        print(f"Error: {discord_file} not found!")
        print("Please run discord_data_cleaner.py first to clean your Discord data.")
        exit(1)
    
    # Mix the data
    try:
        mix_training_data(discord_file, output_file, discord_ratio)
        
        # Preview the results
        if os.path.exists(output_file):
            preview_mixed_data(output_file)
            
            print("\n" + "=" * 50)
            print("Next Steps:")
            print("=" * 50)
            print("1. Update your train.py file to use the mixed data:")
            print(f"   with open('{output_file}', 'r') as f:")
            print("       training_text = f.read()")
            print("\n2. Run your training script:")
            print("   python train.py")
            print("\n3. The model should now generate much better text!")
            print("=" * 50)
    
    except Exception as e:
        print(f"\nError during mixing: {e}")
        print("\nCreating fallback dataset with built-in quality text...")
        
        with open(discord_file, 'r', encoding='utf-8') as f:
            discord_text = f.read()
        
        fallback = create_fallback_quality_text('data/fallback_quality.txt')
        
        # Mix with fallback
        with open('data/fallback_quality.txt', 'r') as f:
            quality_text = f.read()
        
        discord_paras = [p for p in discord_text.split('\n\n') if p.strip()]
        quality_paras = [p for p in quality_text.split('\n\n') if p.strip()]
        
        all_paras = discord_paras + quality_paras * 3  # More quality text
        random.shuffle(all_paras)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_paras))
        
        print(f"Fallback dataset created: {output_file}")