import discord
from discord import app_commands
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleLLM
from tokenizer import WordTokenizer
from config import Config
import os
import asyncio
import re
import aiohttp
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import requests

load_dotenv()

class GrammarEnhancer:
    """Enhances text with proper grammar and punctuation"""
    
    @staticmethod
    def detect_question(text):
        """Detect if text is a question"""
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom', 
                         'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'is', 'are', 'was']
        text_lower = text.lower().strip()
        
        # Check for question mark
        if '?' in text:
            return True
        
        # Check if starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ""
        return first_word in question_words
    
    @staticmethod
    def detect_request(text):
        """Detect if text is a request/command"""
        request_words = ['please', 'could you', 'can you', 'would you', 'will you', 
                        'make', 'create', 'generate', 'show', 'give', 'tell', 'help']
        imperative_verbs = ['make', 'create', 'show', 'tell', 'give', 'send', 'draw', 'edit']
        
        text_lower = text.lower().strip()
        
        # Check for request phrases
        for phrase in request_words:
            if phrase in text_lower:
                return True
        
        # Check if starts with imperative verb
        first_word = text_lower.split()[0] if text_lower.split() else ""
        return first_word in imperative_verbs
    
    @staticmethod
    def fix_capitalization(text):
        """Fix capitalization in text"""
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        # Capitalize I
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
    
    @staticmethod
    def add_punctuation(text):
        """Add proper punctuation"""
        if not text:
            return text
        
        text = text.strip()
        
        # If ends with incomplete sentence, add period
        if text and text[-1] not in '.!?,;:':
            text += '.'
        
        return text
    
    @staticmethod
    def enhance_response(text, is_question=False, is_request=False):
        """Enhance generated text with proper grammar"""
        text = GrammarEnhancer.fix_capitalization(text)
        text = GrammarEnhancer.add_punctuation(text)
        return text

class ImageManipulator:
    """Manipulate and generate images"""
    
    @staticmethod
    async def download_image(url):
        """Download image from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        return Image.open(io.BytesIO(data))
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    @staticmethod
    def apply_effect(image, effect_type):
        """Apply various effects to image"""
        try:
            if effect_type == "blur":
                return image.filter(ImageFilter.BLUR)
            elif effect_type == "sharpen":
                return image.filter(ImageFilter.SHARPEN)
            elif effect_type == "edge":
                return image.filter(ImageFilter.FIND_EDGES)
            elif effect_type == "emboss":
                return image.filter(ImageFilter.EMBOSS)
            elif effect_type == "contour":
                return image.filter(ImageFilter.CONTOUR)
            elif effect_type == "invert":
                return ImageOps.invert(image.convert('RGB'))
            elif effect_type == "grayscale":
                return ImageOps.grayscale(image)
            elif effect_type == "flip":
                return ImageOps.flip(image)
            elif effect_type == "mirror":
                return ImageOps.mirror(image)
            elif effect_type == "brighten":
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(1.5)
            elif effect_type == "darken":
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(0.5)
            elif effect_type == "saturate":
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(2.0)
            else:
                return image
        except Exception as e:
            print(f"Error applying effect: {e}")
            return image
    
    @staticmethod
    def create_gradient(width=512, height=512, color1=(255,0,0), color2=(0,0,255)):
        """Create gradient image"""
        image = Image.new('RGB', (width, height))
        pixels = image.load()
        
        for y in range(height):
            r = int(color1[0] + (color2[0] - color1[0]) * y / height)
            g = int(color1[1] + (color2[1] - color1[1]) * y / height)
            b = int(color1[2] + (color2[2] - color1[2]) * y / height)
            for x in range(width):
                pixels[x, y] = (r, g, b)
        
        return image
    
    @staticmethod
    def to_bytes(image):
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

class LLMBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.tree = app_commands.CommandTree(self)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training settings
        self.training_enabled = True
        self.message_buffer = []
        self.buffer_size = 50
        self.messages_collected = 0
        self.total_trained = 0
        self.learning_rate = 0.0001
        self.save_interval = 100
        self.training_sessions = 0
        
        # Grammar enhancer
        self.grammar = GrammarEnhancer()
        
        # Image manipulator
        self.image_handler = ImageManipulator()
        
    async def setup_hook(self):
        """Load model and sync commands"""
        print("=" * 50)
        print("Loading Enhanced LLM Bot...")
        print("=" * 50)
        
        # Load tokenizer
        try:
            self.tokenizer = WordTokenizer.load(Config.TOKENIZER_SAVE_PATH)
            print(f"✅ Loaded WordTokenizer with {self.tokenizer.vocab_size:,} words")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            return
        
        # Load model
        self.model = SimpleLLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            max_seq_len=Config.MAX_SEQ_LEN
        )
        
        # Load weights
        try:
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
        
        # Sync commands
        try:
            synced = await self.tree.sync()
            print(f"✅ Synced {len(synced)} slash commands")
        except Exception as e:
            print(f"❌ Failed to sync: {e}")
        
        print("\n🎓 Features enabled:")
        print("   • Online learning from messages")
        print("   • Grammar and punctuation enhancement")
        print("   • Question/Request detection")
        print("   • Image viewing and manipulation")
    
    async def train_on_messages(self, messages):
        """Train model on collected messages"""
        if not self.training_enabled or self.model is None:
            return
        
        try:
            training_text = " ".join(messages)
            tokens = self.tokenizer.encode(training_text)
            
            if len(tokens) < Config.MAX_LENGTH:
                return
            
            sequences = []
            for i in range(0, len(tokens) - Config.MAX_LENGTH, Config.MAX_LENGTH // 2):
                seq = tokens[i:i + Config.MAX_LENGTH]
                if len(seq) == Config.MAX_LENGTH:
                    sequences.append(seq)
            
            if not sequences:
                return
            
            self.model.train()
            total_loss = 0
            
            for seq in sequences:
                input_seq = torch.tensor([seq[:-1]]).to(self.device)
                target_seq = torch.tensor([seq[1:]]).to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(input_seq)
                loss = self.criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(sequences)
            self.training_sessions += 1
            self.total_trained += len(messages)
            
            print(f"\n✅ Trained on {len(messages)} messages | Loss: {avg_loss:.4f}")
            
            if self.training_sessions % self.save_interval == 0:
                self.save_checkpoint()
            
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.model.eval()
    
    def save_checkpoint(self):
        """Save checkpoint"""
        try:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_sessions': self.training_sessions,
                'total_trained': self.total_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, 'checkpoints/online_learning.pth')
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            
            print(f"💾 Saved: {self.training_sessions} sessions, {self.total_trained} messages")
            
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def generate_response(self, prompt, max_length=100, temperature=0.8, enhance_grammar=True):
        """Generate enhanced response"""
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded"
        
        try:
            # Detect intent
            is_question = self.grammar.detect_question(prompt)
            is_request = self.grammar.detect_request(prompt)
            
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) == 0:
                tokens = [1]
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(max_length):
                    context = tokens[-Config.MAX_SEQ_LEN:]
                    input_tensor = torch.tensor([context]).to(self.device)
                    output = self.model(input_tensor)
                    
                    logits = output[0, -1, :] / temperature
                    probs = F.softmax(logits, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                    tokens.append(next_token)
                    
                    next_word = self.tokenizer.idx_to_token.get(next_token, '<UNK>')
                    if next_word in ['.', '!', '?'] and len(tokens) > 10:
                        break
            
            response = self.tokenizer.decode(tokens)
            
            # Remove prompt
            prompt_lower = prompt.lower()
            response_lower = response.lower()
            if response_lower.startswith(prompt_lower):
                response = response[len(prompt):].strip()
            
            # Enhance with grammar
            if enhance_grammar and response:
                response = self.grammar.enhance_response(response, is_question, is_request)
            
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "❌ Sorry, I had trouble generating a response."
    
    async def on_ready(self):
        print("\n" + "=" * 50)
        print(f'✅ Bot ready as {self.user}')
        print(f'✅ Connected to {len(self.guilds)} servers')
        print("=" * 50)
    
    async def on_message(self, message):
        # Ignore bots
        if message.author.bot or not message.content:
            return
        
        # Collect for training
        if self.training_enabled and not message.content.startswith('/'):
            self.message_buffer.append(message.content)
            self.messages_collected += 1
            
            if len(self.message_buffer) >= self.buffer_size:
                print(f"\n📚 Training on {len(self.message_buffer)} messages...")
                messages = self.message_buffer.copy()
                self.message_buffer.clear()
                asyncio.create_task(self.train_on_messages(messages))
        
        # Respond to mentions
        if self.user in message.mentions:
            prompt = message.content.replace(f'<@{self.user.id}>', '').strip()
            
            if not prompt:
                await message.channel.send("Hey! Ask me something!")
                return
            
            async with message.channel.typing():
                response = self.generate_response(prompt, max_length=150, temperature=0.8)
                
                if len(response) > 1900:
                    response = response[:1900] + "..."
                
                await message.channel.send(response)

bot = LLMBot()

# Slash Commands
@bot.tree.command(name="generate", description="Generate text with enhanced grammar")
@app_commands.describe(prompt="Text prompt", temperature="Creativity (0.5-1.5)", max_length="Max words")
async def generate(interaction: discord.Interaction, prompt: str, temperature: float = 0.8, max_length: int = 100):
    await interaction.response.defer(thinking=True)
    
    temperature = max(0.3, min(2.0, temperature))
    max_length = max(10, min(300, max_length))
    
    response = bot.generate_response(prompt, max_length, temperature)
    
    if len(response) > 1900:
        response = response[:1900] + "..."
    
    embed = discord.Embed(title="🤖 Generated Text", description=response, color=discord.Color.blue())
    embed.add_field(name="📝 Prompt", value=prompt[:100], inline=False)
    
    # Show detected intent
    is_q = bot.grammar.detect_question(prompt)
    is_r = bot.grammar.detect_request(prompt)
    intent = "Question" if is_q else "Request" if is_r else "Statement"
    embed.add_field(name="🎯 Intent", value=intent, inline=True)
    embed.add_field(name="🌡️ Temp", value=f"{temperature:.2f}", inline=True)
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="view_image", description="View and analyze an image")
@app_commands.describe(url="Image URL or use attachment")
async def view_image(interaction: discord.Interaction, url: str = None):
    await interaction.response.defer(thinking=True)
    
    try:
        # Get image from URL or attachment
        image_url = url
        if not image_url and interaction.message and interaction.message.attachments:
            image_url = interaction.message.attachments[0].url
        
        if not image_url:
            await interaction.followup.send("❌ Please provide an image URL or attachment!")
            return
        
        # Download image
        image = await bot.image_handler.download_image(image_url)
        
        if image:
            embed = discord.Embed(title="🖼️ Image Analyzed", color=discord.Color.green())
            embed.add_field(name="Size", value=f"{image.width}x{image.height}", inline=True)
            embed.add_field(name="Format", value=image.format or "Unknown", inline=True)
            embed.add_field(name="Mode", value=image.mode, inline=True)
            embed.set_image(url=image_url)
            
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("❌ Failed to load image")
            
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

@bot.tree.command(name="edit_image", description="Apply effects to an image")
@app_commands.describe(url="Image URL", effect="Effect to apply")
@app_commands.choices(effect=[
    app_commands.Choice(name="Blur", value="blur"),
    app_commands.Choice(name="Sharpen", value="sharpen"),
    app_commands.Choice(name="Edge Detect", value="edge"),
    app_commands.Choice(name="Emboss", value="emboss"),
    app_commands.Choice(name="Invert", value="invert"),
    app_commands.Choice(name="Grayscale", value="grayscale"),
    app_commands.Choice(name="Flip", value="flip"),
    app_commands.Choice(name="Mirror", value="mirror"),
    app_commands.Choice(name="Brighten", value="brighten"),
    app_commands.Choice(name="Darken", value="darken"),
])
async def edit_image(interaction: discord.Interaction, url: str, effect: str):
    await interaction.response.defer(thinking=True)
    
    try:
        image = await bot.image_handler.download_image(url)
        
        if not image:
            await interaction.followup.send("❌ Failed to download image")
            return
        
        # Apply effect
        edited = bot.image_handler.apply_effect(image, effect)
        
        # Convert to bytes
        buffer = bot.image_handler.to_bytes(edited)
        
        # Send
        file = discord.File(buffer, filename=f"edited_{effect}.png")
        embed = discord.Embed(title=f"🎨 Applied: {effect.title()}", color=discord.Color.purple())
        embed.set_image(url=f"attachment://edited_{effect}.png")
        
        await interaction.followup.send(embed=embed, file=file)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

@bot.tree.command(name="create_gradient", description="Create a gradient image")
@app_commands.describe(color1="First color (hex)", color2="Second color (hex)")
async def create_gradient(interaction: discord.Interaction, color1: str = "#FF0000", color2: str = "#0000FF"):
    await interaction.response.defer(thinking=True)
    
    try:
        # Parse hex colors
        c1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create gradient
        image = bot.image_handler.create_gradient(512, 512, c1, c2)
        buffer = bot.image_handler.to_bytes(image)
        
        file = discord.File(buffer, filename="gradient.png")
        embed = discord.Embed(title="🌈 Gradient Created", color=discord.Color.gold())
        embed.set_image(url="attachment://gradient.png")
        
        await interaction.followup.send(embed=embed, file=file)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

@bot.tree.command(name="status", description="Check bot status")
async def status(interaction: discord.Interaction):
    embed = discord.Embed(title="🤖 Bot Status", color=discord.Color.green())
    
    if bot.model:
        embed.add_field(name="Model", value="✅ Loaded", inline=True)
        embed.add_field(name="Training", value="✅ Active" if bot.training_enabled else "⏸️ Paused", inline=True)
        embed.add_field(name="Buffer", value=f"{len(bot.message_buffer)}/{bot.buffer_size}", inline=True)
        embed.add_field(name="Grammar", value="✅ Enhanced", inline=True)
        embed.add_field(name="Images", value="✅ Enabled", inline=True)
        embed.add_field(name="Device", value=bot.device.upper(), inline=True)
    else:
        embed.description = "❌ Model not loaded"
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="training", description="Toggle training")
async def training(interaction: discord.Interaction):
    bot.training_enabled = not bot.training_enabled
    status = "✅ Enabled" if bot.training_enabled else "⏸️ Disabled"
    
    embed = discord.Embed(title="🎓 Training", description=f"Now **{status}**", 
                         color=discord.Color.green() if bot.training_enabled else discord.Color.orange())
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="stats", description="Training statistics")
async def stats(interaction: discord.Interaction):
    embed = discord.Embed(title="📊 Statistics", color=discord.Color.blue())
    embed.add_field(name="Sessions", value=str(bot.training_sessions), inline=True)
    embed.add_field(name="Messages Trained", value=str(bot.total_trained), inline=True)
    embed.add_field(name="Collected", value=str(bot.messages_collected), inline=True)
    embed.add_field(name="Buffer", value=f"{len(bot.message_buffer)}/{bot.buffer_size}", inline=True)
    embed.add_field(name="Learning Rate", value=str(bot.learning_rate), inline=True)
    embed.add_field(name="Status", value="✅ Active" if bot.training_enabled else "⏸️ Paused", inline=True)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="save", description="Save model checkpoint")
async def save(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    bot.save_checkpoint()
    
    embed = discord.Embed(title="💾 Saved", description="Checkpoint saved!", color=discord.Color.green())
    embed.add_field(name="Sessions", value=str(bot.training_sessions), inline=True)
    embed.add_field(name="Messages", value=str(bot.total_trained), inline=True)
    
    await interaction.followup.send(embed=embed)

if __name__ == "__main__":
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not TOKEN:
        print("\n❌ DISCORD_BOT_TOKEN not found in .env!")
        exit(1)
    
    print("=" * 50)
    print("Enhanced Discord LLM Bot")
    print("=" * 50)
    print("\n✨ Features:")
    print("  • Online learning from messages")
    print("  • Grammar & punctuation enhancement")
    print("  • Question/Request detection")
    print("  • Image viewing & manipulation")
    print("  • Gradient generation")
    print("\n⚡ Starting...")
    print("=" * 50 + "\n")
    
    bot.run(TOKEN)