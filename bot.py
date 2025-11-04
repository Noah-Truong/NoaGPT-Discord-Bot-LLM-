import discord
from discord.ext import commands
import torch
import torch.nn.functional as F
from model import SimpleLLM
from tokenizer import CharTokenizer
from config import Config
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        
    async def setup_hook(self):
        """Load model when bot starts"""
        print("Loading model...")
        
        # Load tokenizer
        self.tokenizer = CharTokenizer.load(Config.TOKENIZER_SAVE_PATH)
        
        # Load model
        self.model = SimpleLLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            max_seq_len=Config.MAX_SEQ_LEN
        )
        
        # Load trained weights
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
    
    def generate_response(self, prompt, max_length=150, temperature=0.8):
        """Generate text from prompt"""
        tokens = self.tokenizer.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Limit context window
                context = tokens[-Config.MAX_SEQ_LEN:]
                input_tensor = torch.tensor([context]).to(self.device)
                output = self.model(input_tensor)
                
                # Get next token
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
                
                # Stop at newline for more natural responses
                if self.tokenizer.decode([next_token]) == '\n':
                    break
        
        response = self.tokenizer.decode(tokens)
        # Return only the generated part (remove prompt)
        return response[len(prompt):]
    
    async def on_ready(self):
        print(f'✅ Bot logged in as {self.user}')
        print(f'✅ Ready to chat in {len(self.guilds)} servers')
    
    async def on_message(self, message):
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Process commands first
        await self.process_commands(message)
        
        # Respond when mentioned or in DMs
        if self.user in message.mentions or isinstance(message.channel, discord.DMChannel):
            # Remove bot mention from message
            prompt = message.content.replace(f'<@{self.user.id}>', '').strip()
            
            if not prompt:
                await message.channel.send("Hey! Ask me something!")
                return
            
            # Show typing indicator
            async with message.channel.typing():
                try:
                    # Generate response
                    response = self.generate_response(prompt, max_length=200, temperature=0.8)
                    
                    # Clean up response
                    response = response.strip()
                    
                    # Limit response length (Discord has 2000 char limit)
                    if len(response) > 1900:
                        response = response[:1900] + "..."
                    
                    if response:
                        await message.channel.send(response)
                    else:
                        await message.channel.send("🤔 I'm not sure how to respond to that...")
                
                except Exception as e:
                    print(f"Error generating response: {e}")
                    await message.channel.send("❌ Sorry, I had trouble generating a response.")

# Commands
@commands.command(name='generate')
async def generate_command(ctx, *, prompt):
    """Generate text from a prompt. Usage: !generate <prompt>"""
    async with ctx.typing():
        try:
            response = ctx.bot.generate_response(prompt, max_length=300, temperature=0.9)
            response = response.strip()
            
            if len(response) > 1900:
                response = response[:1900] + "..."
            
            if response:
                await ctx.send(f"**Generated:**\n{response}")
            else:
                await ctx.send("❌ Couldn't generate text from that prompt.")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

@commands.command(name='temperature')
async def temperature_command(ctx, temp: float, *, prompt):
    """Generate with custom temperature. Usage: !temperature 0.5 <prompt>"""
    if temp < 0.1 or temp > 2.0:
        await ctx.send("❌ Temperature must be between 0.1 and 2.0")
        return
    
    async with ctx.typing():
        try:
            response = ctx.bot.generate_response(prompt, max_length=200, temperature=temp)
            response = response.strip()
            
            if len(response) > 1900:
                response = response[:1900] + "..."
            
            await ctx.send(f"**Temperature {temp}:**\n{response}")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

@commands.command(name='status')
async def status_command(ctx):
    """Check bot status"""
    embed = discord.Embed(title="🤖 Bot Status", color=discord.Color.green())
    embed.add_field(name="Model", value="SimpleLLM", inline=True)
    embed.add_field(name="Device", value=ctx.bot.device, inline=True)
    embed.add_field(name="Vocab Size", value=ctx.bot.tokenizer.vocab_size, inline=True)
    embed.add_field(name="Servers", value=len(ctx.bot.guilds), inline=True)
    await ctx.send(embed=embed)

# Setup bot
bot = LLMBot()
bot.add_command(generate_command)
bot.add_command(temperature_command)
bot.add_command(status_command)

if __name__ == "__main__":
    # Load token from .env file
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not TOKEN:
        print("\n❌ Error: DISCORD_BOT_TOKEN not found in .env file!")
        print("Create a .env file with: DISCORD_BOT_TOKEN=your_token_here")
        exit(1)
    
    print("=" * 50)
    print("Starting Discord LLM Bot")
    print("=" * 50)
    print("\n📝 Setup Instructions:")
    print("1. Make sure you've trained your model first")
    print("2. Create a bot at: https://discord.com/developers/applications")
    print("3. Enable 'Message Content Intent' in Bot settings")
    print("4. Copy your bot token and paste it above")
    print("5. Invite bot to your server\n")
    print("Commands:")
    print("  - Mention the bot to chat")
    print("  - !generate <prompt> - Generate text")
    print("  - !temperature <0.1-2.0> <prompt> - Generate with custom temp")
    print("  - !status - Check bot status")
    print("=" * 50 + "\n")
    
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        print("\n❌ Invalid bot token! Please update TOKEN in the script.")
    except Exception as e:
        print(f"\n❌ Error: {e}")