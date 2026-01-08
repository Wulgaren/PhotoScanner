# PhotoScanner 📸

AI-powered photo curation assistant that learns your preferences from your Apple Photos library.

## Features

- 🎯 **Learns your taste** - Trains on your manually curated photos
- 📊 **Quality scoring** - Rates photos based on your preferences
- 🔗 **Series detection** - Groups similar photos and bursts together
- 🛡️ **Safe suggestions** - Never suggests deleting the best photo in a series, always keeps at least one
- 🍎 **Apple Photos integration** - Works directly with your Photos library

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Analyze your library
```bash
python analyze_library.py
```

### 2. Train the model on your curated photos
```bash
python train_model.py --cutoff-date 2023-11-18
```

### 3. Scan new photos for suggestions
```bash
python scan_photos.py --after 2023-11-18
```

## How It Works

1. **Feature Extraction**: Uses a pretrained EfficientNet to extract visual features
2. **Preference Learning**: Learns what makes a photo "good" based on your curated collection
3. **Series Grouping**: Groups photos by time and visual similarity
4. **Smart Ranking**: Within each series, ranks photos by predicted quality
5. **Safe Suggestions**: Only suggests deletions when better alternatives exist

## Privacy

All processing happens locally on your Mac. No photos are uploaded anywhere.
