import discord
import json
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class MessageScraper(discord.Client):
    def __init__(self, channel_ids, output_file='data/training_data.txt'):
        super().__init__(intents=discord.Intents.default())
        self.channel_ids = channel_ids
        self.output_file = output_file
        self.messages = []
    
    async def on_ready(self):
        print(f'✅ Logged in as {self.user}')
        print(f'📥 Starting to scrape messages...\n')
        
        for channel_id in self.channel_ids:
            channel = self.get_channel(channel_id)
            
            if channel is None:
                print(f"⚠️  Could not access channel {channel_id}")
                continue
            
            print(f"📝 Scraping #{channel.name}...")
            count = 0
            
            try:
                async for message in channel.history(limit=None):
                    # Skip bot messages and empty messages
                    if not message.author.bot and message.content.strip():
                        formatted_msg = f"{message.author.name}: {message.content}"
                        self.messages.append(formatted_msg)
                        count += 1
                        
                        # Progress indicator
                        if count % 100 == 0:
                            print(f"   Scraped {count} messages...")
                
                print(f"✅ Completed #{channel.name}: {count} messages\n")
            
            except discord.Forbidden:
                print(f"❌ No permission to read #{channel.name}\n")
            except Exception as e:
                print(f"❌ Error scraping #{channel.name}: {e}\n")
        
        # Save all messages
        self.save_messages()
        await self.close()
    
    def save_messages(self):
        """Save messages to training file"""
        if not self.messages:
            print("❌ No messages scraped!")
            return
        
        # Create data directory if needed
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save as text file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.messages))
        
        # Also save raw JSON for reference
        json_file = self.output_file.replace('.txt', '_raw.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Scraping complete!")
        print(f"✅ {len(self.messages)} total messages saved")
        print(f"✅ Training data: {self.output_file}")
        print(f"✅ Raw JSON: {json_file}")
        print(f"✅ File size: {os.path.getsize(self.output_file) / 1024 / 1024:.2f} MB")

async def main():
    # Configuration
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    # List of channel IDs to scrape (right-click channel → Copy ID)
    CHANNEL_IDS = [
        1281071460244263007,
        1409321509591322755
    ]
    
    client = MessageScraper(CHANNEL_IDS)
    await client.start(TOKEN)

if __name__ == "__main__":
    print("=" * 50)
    print("Discord Message Scraper")
    print("=" * 50)
    print("\n⚠️  IMPORTANT:")
    print("   1. Create a bot at: https://discord.com/developers/applications")
    print("   2. Enable 'Message Content Intent' in Bot settings")
    print("   3. Invite bot to your server with 'Read Message History' permission")
    print("   4. Enable Developer Mode in Discord to copy channel IDs")
    print("   5. Update TOKEN and CHANNEL_IDS in this script\n")
    
    asyncio.run(main())