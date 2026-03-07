# PhotoScanner 📸

AI-powered photo curation assistant that learns your preferences from your Apple Photos library. Also includes a Twitter/X image curator that uses the same trained model!

## Features

- 🎯 **Learns your taste** - Trains on your manually curated photos
- 📊 **Quality scoring** - Rates photos based on your preferences
- 🔗 **Series detection** - Groups similar photos and bursts together
- 🛡️ **Safe suggestions** - Never suggests deleting the best photo in a series
- 🍎 **Apple Photos integration** - Works directly with your Photos library
- 🔄 **Feedback learning** - Improves over time based on your corrections
- 🐦 **Twitter curator** - Auto-curates images from Twitter/X using the same model
- 🔒 **100% Private** - All processing happens locally on your Mac

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
# Train on your favorited photos before a cutoff date
python train_model.py --cutoff-date 2023-11-18
```

**Optional: Add bad examples for better accuracy**

Create a `BadPhotos/` folder and add photos you consider bad. The model will learn to distinguish good from bad:

```bash
mkdir BadPhotos
# Copy/move bad example photos into BadPhotos/
python train_model.py --cutoff-date 2023-11-18
```

### 2. Scan photos for suggestions

```bash
# Scan favorited photos after the cutoff date
python scan_photos.py --after 2023-11-18 --threshold 0.8
```

### 3. Review and delete

**Option A: Move to album (recommended)**

```bash
# Move low-scoring photos to "To Delete" album in Photos app
python move_to_album.py --threshold 0.8

# Dry run first to see what would be moved
python move_to_album.py --threshold 0.8 --dry-run
```

Then open Photos app → "To Delete" album → review → delete what you don't want.

**Option B: Interactive review**

```bash
python interactive_review.py
```

Commands:
- `v` - View all photos in Preview
- `v #` - View specific photo (e.g., `v 2`)
- `f` - Show in Finder
- `k #` - Keep only photo # (delete others)
- `d #` - Delete specific photos (e.g., `d 2 3`)
- `da` - Delete ALL in this series
- `y` - Confirm suggestions
- `n` - Keep all
- `s` - Skip
- `q` - Quit

### 4. Learn from feedback

After sorting through the "To Delete" album:

```bash
python learn_from_feedback.py
```

This checks which photos you:
- **Rescued** (removed from album but kept in library) → become positive examples
- **Left in album or deleted** → become negative examples

Then retrain:

```bash
python train_model.py --cutoff-date 2023-11-18
```

The model gets smarter with each feedback cycle! 🧠

---

## Twitter Curator 🐦

Automatically curate images from Twitter/X using your trained aesthetic model.

### Setup

1. **Create a Discord bot** at [Discord Developer Portal](https://discord.com/developers/applications)
   - Create new application → Bot section → Reset Token → Copy
   - Enable **"Message Content Intent"** under Privileged Gateway Intents
   - OAuth2 → URL Generator → Select `bot` scope + `Read Messages/View Channels` + `Read Message History`
   - Add bot to your server

2. **Get channel IDs** (Enable Developer Mode in Discord → Right-click channel → Copy ID)

3. **Edit `config.json`**:
```json
{
    "discord_token": "your-bot-token",
    "tweetshift_channel_ids": [
        123456789012345678,
        123456789012345679
    ],
    "score_threshold": 0.8
}
```

### Usage

```bash
# Fetch last 24 hours and exit
python twitter_curator.py --hours 24 --no-listen

# Fetch last 24 hours, then keep listening for new tweets
python twitter_curator.py --hours 24

# Just listen for new tweets (no backfill)
python twitter_curator.py
```

### Output

Images are saved to `~/Pictures/TwitterImages/`:

| Folder | Contents |
|--------|----------|
| `all/` | Non-curated images only (curated images stay in `curated/` only) |
| `videos/` | Non-curated videos/GIFs only (curated videos stay in `curated/` only) |
| `curated/` | High-scoring images (≥ threshold) and videos whose tweet text has keywords |
| `announcements/` | Images from tweets with announcement keywords |

- Filenames include the Twitter author: `AuthorName_20260108_143022_abc123.jpg`
- Tweet URL is embedded in image EXIF metadata
- `announcements/announcements.txt` logs important tweets

---

## How It Works

1. **Feature Extraction**: Uses pretrained EfficientNet to extract visual features
2. **Preference Learning**: Learns your aesthetic preferences from curated photos
3. **Series Grouping**: Groups photos by time and visual similarity (perceptual hashing)
4. **Smart Ranking**: Ranks photos by predicted quality within each series
5. **Feedback Loop**: Continuously improves based on your corrections

## Privacy

✅ All processing happens locally on your Mac  
✅ No photos are uploaded anywhere  
✅ Model weights stay on your device  
✅ Discord bot only reads messages, doesn't store externally

## Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train the aesthetic model |
| `scan_photos.py` | Scan photos and generate suggestions |
| `move_to_album.py` | Move low-scoring photos to Photos album |
| `interactive_review.py` | Interactively review suggestions |
| `learn_from_feedback.py` | Learn from your sorting decisions |
| `twitter_curator.py` | Discord bot for Twitter image curation |
| `config.json` | Configuration (Discord token, channels, etc.) |

## Tips

- **Start with a high threshold** (0.8) and lower it if needed
- **Add bad examples** to `BadPhotos/` for much better accuracy
- **Run feedback learning** after each sorting session
- **Safe to interrupt** - progress is saved incrementally during training
