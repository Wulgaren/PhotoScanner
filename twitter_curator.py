#!/usr/bin/env python3
"""
Discord bot that monitors TweetShift channels and curates images using the trained model.
Saves all images, curates the good ones, and flags important announcements.
"""

import discord
import aiohttp
import asyncio
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from io import BytesIO
import hashlib
import piexif

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from photo_scanner.feature_extractor import FeatureExtractor, AestheticScorer

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ============ CONFIGURATION ============
# Edit config.json to set your settings!

CONFIG_PATH = Path(__file__).parent / 'config.json'

def load_config():
    """Load configuration from config.json"""
    if not CONFIG_PATH.exists():
        print(f"❌ Config file not found: {CONFIG_PATH}")
        return None
    
    with open(CONFIG_PATH) as f:
        return json.load(f)

# Load config
_config = load_config()

if _config:
    DISCORD_TOKEN = _config.get('discord_token', 'YOUR_BOT_TOKEN_HERE')
    TWEETSHIFT_CHANNEL_IDS = _config.get('tweetshift_channel_ids', [])
    SCORE_THRESHOLD = _config.get('score_threshold', 0.7)
    
    save_dir = _config.get('save_directory', '~/Pictures/TwitterImages')
    BASE_SAVE_DIR = Path(save_dir).expanduser()
    
    ANNOUNCEMENT_KEYWORDS = _config.get('announcement_keywords', [
        "announce", "release", "drop", "launch", "available now"
    ])
    ALWAYS_CURATE_ACCOUNTS = _config.get('always_curate_accounts', [])
else:
    DISCORD_TOKEN = "YOUR_BOT_TOKEN_HERE"
    TWEETSHIFT_CHANNEL_IDS = []
    SCORE_THRESHOLD = 0.7
    BASE_SAVE_DIR = Path.home() / "Pictures" / "TwitterImages"
    ANNOUNCEMENT_KEYWORDS = ["announce", "release", "drop", "launch"]
    ALWAYS_CURATE_ACCOUNTS = []

ALL_IMAGES_DIR = BASE_SAVE_DIR / "all"
CURATED_DIR = BASE_SAVE_DIR / "curated"
ANNOUNCEMENTS_DIR = BASE_SAVE_DIR / "announcements"
VIDEOS_DIR = BASE_SAVE_DIR / "videos"

# ==========================================

CACHE_DIR = Path(__file__).parent / '.cache'
MODEL_PATH = CACHE_DIR / 'aesthetic_model.pkl'


class ImageScorer:
    """Handles image scoring using the trained model."""
    
    def __init__(self):
        self.extractor = None
        self.scorer = None
        self.loaded = False
    
    def load(self):
        """Load the model (called once on startup)."""
        if self.loaded:
            return True
            
        if not MODEL_PATH.exists():
            print(f"❌ Model not found at {MODEL_PATH}")
            print("   Run train_model.py first!")
            return False
        
        print("Loading aesthetic model...")
        self.extractor = FeatureExtractor(model_name='efficientnet_b0')
        self.scorer = AestheticScorer(self.extractor)
        self.scorer.load(MODEL_PATH)
        self.loaded = True
        print("✓ Model loaded!")
        return True
    
    def score_image(self, image_bytes: bytes) -> float:
        """Score an image from bytes. Returns 0.0-1.0."""
        if not self.loaded:
            return 0.5
        
        try:
            from PIL import Image
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Save temporarily and extract features
            temp_path = CACHE_DIR / 'temp_score.jpg'
            img.save(temp_path, 'JPEG')
            
            features = self.extractor.extract_single(temp_path)
            if features is None:
                return 0.5
            
            score = self.scorer.score(features.reshape(1, -1))[0]
            temp_path.unlink()  # Clean up
            
            return float(score)
        except Exception as e:
            print(f"Error scoring image: {e}")
            return 0.5


def is_announcement(text: str) -> bool:
    """Check if text contains announcement keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ANNOUNCEMENT_KEYWORDS)


def is_always_curate_account(author: str) -> bool:
    """Check if author matches any always-curate account keywords."""
    if not author or not ALWAYS_CURATE_ACCOUNTS:
        return False
    author_lower = author.lower()
    return any(keyword.lower() in author_lower for keyword in ALWAYS_CURATE_ACCOUNTS)


def extract_tweet_info(message: discord.Message) -> dict:
    """Extract tweet URL and text from a TweetShift message."""
    info = {
        'text': '',
        'tweet_url': None,
        'author': None,
        'username': None,  # Twitter username from URL
    }
    
    # Get author from message poster's display name (TweetShift changes its name to the tweet author)
    if message.author:
        author = message.author.display_name
        # Strip "• TweetShift" suffix that TweetShift adds
        author = re.sub(r'\s*[•·]\s*TweetShift\s*$', '', author, flags=re.IGNORECASE)
        info['author'] = author.strip()
    
    # Get text from message content
    if message.content:
        info['text'] = message.content
        
        # Try to find Twitter/X URL
        url_pattern = r'https?://(?:twitter\.com|x\.com)/(\w+)/status/\d+'
        matches = re.findall(url_pattern, message.content)
        if matches:
            info['tweet_url'] = re.search(r'https?://(?:twitter\.com|x\.com)/\w+/status/\d+', message.content).group(0)
            info['username'] = matches[0]  # Extract username from URL
    
    # Also check embeds for text
    for embed in message.embeds:
        if embed.description:
            info['text'] += '\n' + embed.description
        if embed.url and ('twitter.com' in embed.url or 'x.com' in embed.url):
            info['tweet_url'] = embed.url
            # Extract username from embed URL
            username_match = re.search(r'https?://(?:twitter\.com|x\.com)/(\w+)/status/\d+', embed.url)
            if username_match:
                info['username'] = username_match.group(1)
    
    return info


def get_image_urls(message: discord.Message) -> list:
    """Extract image URLs from a message."""
    urls = []
    
    # From attachments
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith('image'):
            # Exclude GIFs - they go to videos folder
            if 'gif' not in attachment.content_type.lower():
                urls.append(attachment.url)
    
    # From embeds
    for embed in message.embeds:
        if embed.image and embed.image.url:
            # Exclude GIFs
            if not embed.image.url.lower().endswith('.gif'):
                urls.append(embed.image.url)
        if embed.thumbnail and embed.thumbnail.url:
            # Skip small thumbnails (usually profile pics)
            if embed.thumbnail.width and embed.thumbnail.width > 100:
                if not embed.thumbnail.url.lower().endswith('.gif'):
                    urls.append(embed.thumbnail.url)
    
    return urls


def get_video_urls(message: discord.Message) -> list:
    """Extract video and GIF URLs from a message."""
    urls = []
    
    # From attachments
    for attachment in message.attachments:
        if attachment.content_type:
            # Videos
            if attachment.content_type.startswith('video'):
                urls.append(attachment.url)
            # GIFs
            elif 'gif' in attachment.content_type.lower():
                urls.append(attachment.url)
    
    # From embeds
    for embed in message.embeds:
        # Video embeds
        if embed.video and embed.video.url:
            urls.append(embed.video.url)
        
        # GIFs in image embeds
        if embed.image and embed.image.url:
            if embed.image.url.lower().endswith('.gif'):
                urls.append(embed.image.url)
        
        # GIFs in thumbnails
        if embed.thumbnail and embed.thumbnail.url:
            if embed.thumbnail.url.lower().endswith('.gif'):
                urls.append(embed.thumbnail.url)
    
    return urls


async def download_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """Download an image from URL."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None


def add_metadata_to_image(image_data: bytes, tweet_url: str, author: str = None) -> bytes:
    """Add tweet URL and author to image EXIF metadata."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(image_data))
        
        # Build description
        description = ""
        if author:
            description += f"Author: {author}\n"
        if tweet_url:
            description += f"Source: {tweet_url}"
        
        # Only works well with JPEG
        if img.format == 'JPEG' or image_data[:2] == b'\xff\xd8':
            try:
                # Try to load existing EXIF
                exif_dict = piexif.load(image_data)
            except:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # Add description to ImageDescription tag
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
            
            # Convert back to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save with new EXIF
            output = BytesIO()
            img.save(output, format='JPEG', exif=exif_bytes, quality=95)
            return output.getvalue()
        else:
            # For non-JPEG, try PNG info
            from PIL import PngImagePlugin
            if img.format == 'PNG' or image_data[:4] == b'\x89PNG':
                output = BytesIO()
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("Description", description)
                pnginfo.add_text("Source", tweet_url or "")
                pnginfo.add_text("Author", author or "")
                img.save(output, format='PNG', pnginfo=pnginfo)
                return output.getvalue()
        
        # Return original if we can't add metadata
        return image_data
        
    except Exception as e:
        print(f"Could not add metadata: {e}")
        return image_data


def generate_filename(url: str, tweet_url: str = None, author: str = None, username: str = None) -> str:
    """Generate a unique filename for an image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get original extension
    ext = '.jpg'
    if '.' in url.split('/')[-1].split('?')[0]:
        ext = '.' + url.split('/')[-1].split('?')[0].split('.')[-1]
    
    # Create a short hash for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Prefer username over display name for filename
    # If username not available, try to extract from tweet_url
    name_to_use = username
    if not name_to_use and tweet_url:
        username_match = re.search(r'https?://(?:twitter\.com|x\.com)/(\w+)/status/\d+', tweet_url)
        if username_match:
            name_to_use = username_match.group(1)
    
    # Fall back to display name if username not available
    if not name_to_use:
        name_to_use = author
    
    # Include name as first part of filename
    name_part = ""
    if name_to_use:
        # Clean name for filename (username should already be clean, but just in case)
        name_clean = re.sub(r'[^\w\-]', '', name_to_use.replace(' ', '_'))[:20]
        name_part = f"{name_clean}_"
    
    return f"{name_part}{timestamp}_{url_hash}{ext}"


class TwitterCurator(discord.Client):
    """Discord bot that curates Twitter images."""
    
    def __init__(self, backfill_hours: int = 0, exit_after_backfill: bool = False):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.scorer = ImageScorer()
        self.session = None
        self.backfill_hours = backfill_hours
        self.exit_after_backfill = exit_after_backfill
        self.processed_ids = set()  # Track processed messages to avoid duplicates
        self.stats = {
            'images_seen': 0,
            'images_curated': 0,
            'announcements': 0,
            'videos_saved': 0,
            'videos_curated': 0,
        }
    
    async def setup_hook(self):
        """Called when bot is starting up."""
        self.session = aiohttp.ClientSession()
        
        # Create directories
        for dir_path in [ALL_IMAGES_DIR, CURATED_DIR, ANNOUNCEMENTS_DIR, VIDEOS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.scorer.load()
    
    async def close(self):
        """Clean up on shutdown."""
        if self.session:
            await self.session.close()
        await super().close()
    
    async def on_ready(self):
        print(f'\n{"="*50}')
        print(f'🐦 Twitter Curator Bot is running!')
        print(f'{"="*50}')
        print(f'Logged in as: {self.user}')
        print(f'Monitoring {len(TWEETSHIFT_CHANNEL_IDS)} channels')
        print(f'Score threshold: {SCORE_THRESHOLD}')
        if ALWAYS_CURATE_ACCOUNTS:
            print(f'Always curate accounts: {", ".join(ALWAYS_CURATE_ACCOUNTS)}')
        print(f'\nSaving to:')
        print(f'  All images: {ALL_IMAGES_DIR}')
        print(f'  Curated:    {CURATED_DIR}')
        print(f'  Videos/GIFs: {VIDEOS_DIR}')
        print(f'  Announcements: {ANNOUNCEMENTS_DIR}')
        print(f'{"="*50}\n')
        
        # Backfill recent messages if requested
        if self.backfill_hours > 0:
            await self.backfill_messages()
            
            if self.exit_after_backfill:
                print("\n✅ Backfill complete. Exiting (--no-listen mode)")
                await self.close()
    
    async def backfill_messages(self):
        """Fetch and process messages from the last N hours."""
        print(f'\n📥 Backfilling messages from the last {self.backfill_hours} hours...\n')
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.backfill_hours)
        total_messages = 0
        
        for channel_id in TWEETSHIFT_CHANNEL_IDS:
            channel = self.get_channel(channel_id)
            if not channel:
                print(f"  ⚠️ Could not find channel {channel_id}")
                continue
            
            print(f'  Fetching from #{channel.name}...')
            channel_count = 0
            
            try:
                async for message in channel.history(after=cutoff_time, limit=None):
                    if message.id not in self.processed_ids:
                        self.processed_ids.add(message.id)
                        await self.process_message(message)
                        channel_count += 1
                        
                        # Small delay to avoid rate limits
                        if channel_count % 10 == 0:
                            await asyncio.sleep(0.5)
                
                print(f'    ✓ Processed {channel_count} messages')
                total_messages += channel_count
                
            except discord.Forbidden:
                print(f"    ❌ No permission to read #{channel.name}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        print(f'\n✓ Backfill complete! Processed {total_messages} messages')
        print(f'  Images seen: {self.stats["images_seen"]}')
        print(f'  Images curated: {self.stats["images_curated"]}')
        print(f'  Videos/GIFs saved: {self.stats["videos_saved"]} (to videos/; curated when keywords in tweet text)')
        print(f'  Announcements: {self.stats["announcements"]}')
        print(f'\n👀 Now listening for new messages...\n')
    
    async def on_message(self, message: discord.Message):
        # Ignore our own messages
        if message.author == self.user:
            return
        
        # Only process messages from monitored channels
        if message.channel.id not in TWEETSHIFT_CHANNEL_IDS:
            return
        
        # Skip if already processed (from backfill)
        if message.id in self.processed_ids:
            return
        
        self.processed_ids.add(message.id)
        await self.process_message(message)
    
    async def process_message(self, message: discord.Message):
        """Process a single message (used by both backfill and live)."""
        # Extract tweet info
        tweet_info = extract_tweet_info(message)
        
        # Get image URLs
        image_urls = get_image_urls(message)
        
        # Get video/GIF URLs
        video_urls = get_video_urls(message)
        
        # If we have images/videos but no tweet URL, check the previous message
        # (TweetShift often sends the tweet link first, then photos in a separate message)
        if (image_urls or video_urls) and not tweet_info.get('tweet_url'):
            try:
                # Look at the previous message in the channel
                async for prev_message in message.channel.history(limit=1, before=message):
                    prev_info = extract_tweet_info(prev_message)
                    if prev_info.get('tweet_url'):
                        # Use tweet info from previous message
                        tweet_info['tweet_url'] = prev_info['tweet_url']
                        tweet_info['username'] = prev_info.get('username')
                        # Keep the author from current message if available, otherwise use previous
                        if not tweet_info.get('author') and prev_info.get('author'):
                            tweet_info['author'] = prev_info['author']
                        # Merge text if needed
                        if prev_info.get('text') and not tweet_info.get('text'):
                            tweet_info['text'] = prev_info['text']
                        break
            except Exception as e:
                # If we can't fetch previous message, continue with what we have
                pass
        
        # Check for announcement
        is_important = is_announcement(tweet_info['text'])
        if is_important:
            self.stats['announcements'] += 1
            print(f"📢 ANNOUNCEMENT detected: {tweet_info['text'][:100]}...")
        
        if not image_urls and not video_urls:
            # No media, but might be important text
            if is_important:
                await self.save_announcement(tweet_info, message)
            return
        
        # Process each image
        for url in image_urls:
            await self.process_image(url, tweet_info, is_important, message)
        
        # Process each video/GIF (no curation, just save)
        for url in video_urls:
            await self.process_video(url, tweet_info)
    
    async def process_image(self, url: str, tweet_info: dict, is_important: bool, message: discord.Message):
        """Download, score, and save an image."""
        # Download
        image_data = await download_image(self.session, url)
        if not image_data:
            return
        
        self.stats['images_seen'] += 1
        
        # Generate filename
        filename = generate_filename(url, tweet_info.get('tweet_url'), tweet_info.get('author'), tweet_info.get('username'))
        
        # Add tweet URL to image metadata
        image_data_with_meta = add_metadata_to_image(
            image_data, 
            tweet_info.get('tweet_url'), 
            tweet_info.get('author')
        )
        
        # Check if author is in always-curate list
        author = tweet_info.get('author')
        always_curate = is_always_curate_account(author)
        
        # Score image (use original data for scoring) - skip if always curating
        if always_curate:
            score = None
        else:
            score = self.scorer.score_image(image_data)
        
        # Curate if good enough or always-curate account
        if always_curate or (score is not None and score >= SCORE_THRESHOLD):
            self.stats['images_curated'] += 1
            curated_path = CURATED_DIR / filename
            curated_path.write_bytes(image_data_with_meta)
            if always_curate:
                print(f"⭐ CURATED (always: {author}): {filename}")
            else:
                print(f"⭐ CURATED ({score:.2f}): {filename}")
        else:
            # Save to "all" only when not curating (curated items stay only in curated/)
            all_path = ALL_IMAGES_DIR / filename
            all_path.write_bytes(image_data_with_meta)
            print(f"   Skipped ({score:.2f}): {filename}")
        
        # Also save to announcements if important
        if is_important:
            announce_path = ANNOUNCEMENTS_DIR / filename
            announce_path.write_bytes(image_data_with_meta)
            # Log announcement text
            self.log_announcement(tweet_info)
    
    async def process_video(self, url: str, tweet_info: dict):
        """Download and save a video or GIF to videos/. Copy to curated/ only if tweet text has announcement keywords."""
        # Download
        video_data = await download_image(self.session, url)  # Same download function works
        if not video_data:
            return
        
        self.stats['videos_saved'] += 1
        
        # Generate filename
        filename = generate_filename(url, tweet_info.get('tweet_url'), tweet_info.get('author'), tweet_info.get('username'))
        
        # Always save to videos folder
        video_path = VIDEOS_DIR / filename
        video_path.write_bytes(video_data)
        
        # Only put in curated/ when tweet text contains announcement keywords
        text = tweet_info.get('text') or ''
        if is_announcement(text):
            self.stats['videos_curated'] += 1
            curated_path = CURATED_DIR / filename
            curated_path.write_bytes(video_data)
            print(f"🎬 VIDEO/GIF saved (curated, has keywords): {filename}")
        else:
            print(f"🎬 VIDEO/GIF saved: {filename}")
    
    def log_announcement(self, tweet_info: dict):
        """Append announcement to the announcements.txt file."""
        announcements_file = ANNOUNCEMENTS_DIR / "announcements.txt"
        
        author = tweet_info.get('author', 'Unknown')
        text = tweet_info.get('text', '').strip()
        tweet_url = tweet_info.get('tweet_url', '')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Clean up text - remove hashtag links like [#TAG](<url>)
        text = re.sub(r'\[#(\w+)\]\(<[^>]+>\)', r'#\1', text)
        # Remove other markdown links [text](<url>)
        text = re.sub(r'\[([^\]]+)\]\(<[^>]+>\)', r'\1', text)
        # Remove duplicate URLs (already have tweet_url)
        text = re.sub(r'https?://twitter\.com/\S+/status/\d+', '', text)
        text = re.sub(r'https?://x\.com/\S+/status/\d+', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        line = f"[{timestamp}] {author}: {text}"
        if tweet_url:
            line += f"\n→ {tweet_url}"
        line += "\n\n"  # Blank line between announcements
        
        with open(announcements_file, 'a') as f:
            f.write(line)
    
    async def save_announcement(self, tweet_info: dict, message: discord.Message):
        """Save an important announcement (text only, no images)."""
        self.log_announcement(tweet_info)
        author = tweet_info.get('author', 'Unknown')
        print(f"📢 Saved announcement from {author}")


def main():
    global SCORE_THRESHOLD
    
    parser = argparse.ArgumentParser(description='Twitter image curator bot')
    parser.add_argument('--hours', type=int, default=0,
                       help='Backfill messages from the last N hours (default: 0, live only)')
    parser.add_argument('--no-listen', action='store_true',
                       help='Exit after backfill instead of listening for new messages')
    parser.add_argument('--threshold', type=float, default=None,
                       help=f'Score threshold for curation (default: {SCORE_THRESHOLD} from config)')
    args = parser.parse_args()
    
    # Override threshold if provided via CLI
    if args.threshold is not None:
        SCORE_THRESHOLD = args.threshold
    
    if DISCORD_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your Discord bot token in config.json")
        print("\nTo get a bot token:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Create a new application")
        print("3. Go to 'Bot' section → Reset Token → Copy")
        print("4. Enable 'Message Content Intent' in Bot settings")
        print("5. Paste token in config.json")
        print("\nTo get channel IDs:")
        print("1. Enable Developer Mode in Discord settings")
        print("2. Right-click channel → Copy ID")
        print("3. Add IDs to config.json tweetshift_channel_ids list")
        return
    
    if not TWEETSHIFT_CHANNEL_IDS:
        print("❌ Please add your TweetShift channel IDs to config.json")
        return
    
    bot = TwitterCurator(backfill_hours=args.hours, exit_after_backfill=args.no_listen)
    
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("\n\n📊 Session Stats:")
        print(f"   Images seen: {bot.stats['images_seen']}")
        print(f"   Images curated: {bot.stats['images_curated']}")
        print(f"   Videos/GIFs saved: {bot.stats['videos_saved']} (to videos/; curated when keywords in tweet text)")
        print(f"   Announcements: {bot.stats['announcements']}")


if __name__ == '__main__':
    main()
