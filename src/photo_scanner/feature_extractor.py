"""Preference model: CLIP backbone + calibrated head on top of frozen features."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import timm
from timm.data import resolve_model_data_config, create_transform

from photo_scanner.paths import CACHE_DIR
from photo_scanner.thumbs import EMBED_MAX_EDGE


# CLIP ViT-B/16: strong semantic + aesthetic features for personal taste learning.
# 224px input stays practical on Apple Silicon (MPS) when scoring thousands of photos.
DEFAULT_BACKBONE = "vit_base_patch16_clip_224.openai"

# Registry of fine-tuned backbones produced by train_finetune.py:
# {"<tag>": {"path": "<.pth>", "base_backbone": "<timm name>", "created": iso}}
FINETUNED_REGISTRY = CACHE_DIR / "finetuned_backbones.json"

FFMPEG = shutil.which("ffmpeg")


def decode_with_ffmpeg(path: Union[str, Path]) -> Optional[Image.Image]:
    """Decode formats Pillow cannot open (e.g. HEIC renamed to .JPG) via ffmpeg.

    Returns an RGB image, or None when ffmpeg is absent or the file is
    undecodable. No temp files: the first frame is piped back as JPEG bytes.
    """
    if not FFMPEG:
        return None
    try:
        result = subprocess.run(
            [
                FFMPEG, "-v", "error", "-i", str(path),
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception:
        return None


def load_rgb_image(path: Union[str, Path]) -> Optional[Image.Image]:
    """Open an image as RGB, falling back to ffmpeg when Pillow rejects it."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as pil_error:
        img = decode_with_ffmpeg(path)
        if img is None:
            print(f"Error loading {path}: {pil_error}")
        return img


def load_finetuned_registry() -> dict:
    if not FINETUNED_REGISTRY.exists():
        return {}
    try:
        data = json.loads(FINETUNED_REGISTRY.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def register_finetuned_backbone(tag: str, path: Union[str, Path], base_backbone: str) -> None:
    registry = load_finetuned_registry()
    registry[tag] = {
        "path": str(path),
        "base_backbone": base_backbone,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")


def resolve_backbone_spec(model_name: str) -> tuple[str, Optional[Path]]:
    """Map a backbone tag to (timm_model_name, optional weights path).

    Fine-tuned tags (registered by train_finetune.py) load custom weights on
    top of their base timm model; everything else is used as-is.
    """
    entry = load_finetuned_registry().get(model_name)
    if entry and Path(entry["path"]).exists():
        return entry["base_backbone"], Path(entry["path"])
    return model_name, None


def load_model_metadata(model_path: Union[str, Path, None] = None) -> dict:
    """Read a saved preference model's metadata without loading the backbone."""
    import pickle

    path = Path(model_path) if model_path else (CACHE_DIR / "aesthetic_model.pkl")
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def recommended_threshold_from_model(model_path: Union[str, Path, None] = None) -> Optional[float]:
    """Read Train's calibrated flag threshold from a model pkl (no backbone load)."""
    value = load_model_metadata(model_path).get("recommended_threshold")
    return float(value) if value is not None else None


def saved_backbone_from_model(model_path: Union[str, Path, None] = None) -> Optional[str]:
    """Backbone tag the model was trained with, so callers do not have to guess."""
    value = load_model_metadata(model_path).get("backbone")
    return str(value) if value else None


class FeatureExtractor:
    """Extract visual features from images using pretrained models."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        device: str = None,
        max_edge: Optional[int] = EMBED_MAX_EDGE,
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: timm backbone, or a fine-tuned tag from the registry
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
            max_edge: Downscale longest edge to this before model transform
                (matches review thumbs; set None to skip)
        """
        self.model_name = model_name
        self.max_edge = max_edge

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Backbone: {self.model_name}")
        if self.max_edge:
            print(f"Max edge before embed: {self.max_edge}")
        
        # Resolve fine-tuned tags to (base timm model, weights)
        base_name, weights_path = resolve_backbone_spec(model_name)

        # Load pretrained model (num_classes=0 → embedding / pooled features)
        self.model = timm.create_model(base_name, pretrained=True, num_classes=0)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state)
            # The fine-tune head is not part of the embedding model; drop it.
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                print(f"  (ignored {len(unexpected)} non-backbone tensors from {weights_path.name})")
            print(f"Loaded fine-tuned weights: {weights_path}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Model-specific preprocessing (CLIP uses different mean/std than ImageNet CNNs)
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        self.input_size = data_config.get('input_size', (3, 224, 224))[-1]
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size).to(self.device)
            self.feature_dim = self.model(dummy).shape[1]
        
        print(f"Feature dimension: {self.feature_dim}")
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        img = load_rgb_image(image_path)
        if img is None:
            return None
        if self.max_edge and max(img.size) > self.max_edge:
            img = img.copy()
            img.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
        return self.transform(img)
    
    @torch.no_grad()
    def extract_single(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract features from a single image."""
        tensor = self.load_image(image_path)
        if tensor is None:
            return None
        
        tensor = tensor.unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, image_paths: list[Union[str, Path]], 
                      batch_size: int = 32,
                      show_progress: bool = True) -> dict:
        """
        Extract features from multiple images.
        
        Returns:
            Dictionary mapping image paths to feature vectors
        """
        results = {}
        
        # Process in batches
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Extracting features", 
                          total=len(image_paths) // batch_size + 1)
        
        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            for path in batch_paths:
                tensor = self.load_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                features = self.model(batch).cpu().numpy()
                
                for path, feat in zip(valid_paths, features):
                    results[str(path)] = feat
        
        return results


def make_preference_head():
    """The classifier that sits on frozen CLIP features.

    An MLP beat RandomForest by ~0.14 AUC on this data (favorites vs BadPhotos,
    80/20 holdout: 0.886 vs 0.746). Kept as a factory so Train's cross-validation
    fits exactly the model that gets saved.
    """
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(512, 128),
        activation='relu',
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42,
    )


class AestheticScorer:
    """
    Score images based on learned aesthetic preferences.
    Uses the extracted features to predict how much the user would like a photo.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.backbone = feature_extractor.model_name
        self.model = None
        self.scaler = None
        self.is_one_class = True
        self.recommended_threshold: Optional[float] = None
    
    def train(self, good_features: np.ndarray, bad_features: np.ndarray = None,
              sample_weight: np.ndarray = None):
        """
        Train the aesthetic scorer.
        
        If bad_features is provided, trains a binary classifier (good vs bad).
        Otherwise, uses one-class learning to identify what "good" looks like.
        
        sample_weight: per-row weights for the binary case (e.g. up-weight
        confirmed-deletion hard negatives).
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        
        self.scaler = StandardScaler()
        good_scaled = self.scaler.fit_transform(good_features)
        
        if bad_features is None or len(bad_features) < 10:
            # One-class classification - learn what "good" looks like
            print(f"Training one-class model on {len(good_features)} curated photos...")
            self.model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
            self.model.fit(good_scaled)
            self.is_one_class = True
        else:
            # Binary classification - learn to distinguish good from bad.
            # MLP beat RandomForest by ~0.14 AUC on this data (see train_metrics.json).
            print(f"Training classifier: {len(good_features)} good, {len(bad_features)} bad...")
            bad_scaled = self.scaler.transform(bad_features)
            
            X = np.vstack([good_scaled, bad_scaled])
            y = np.array([1] * len(good_scaled) + [0] * len(bad_scaled))
            
            self.model = make_preference_head()
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=float)
                if len(sample_weight) != len(X):
                    raise ValueError(
                        f"sample_weight has {len(sample_weight)} rows for {len(X)} samples"
                    )
            self.model.fit(X, y, sample_weight=sample_weight)
            self.is_one_class = False
        
        print("Training complete!")
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Score features. Higher = more similar to your curated photos.
        
        Returns scores between 0 and 1.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        scaled = self.scaler.transform(features.reshape(-1, self.feature_extractor.feature_dim))
        
        if self.is_one_class:
            # One-class SVM returns distance from boundary
            # Positive = inside (similar to training), negative = outside (different)
            raw_scores = self.model.decision_function(scaled)
            # Normalize to 0-1 range using sigmoid
            scores = 1 / (1 + np.exp(-raw_scores))
        else:
            # Binary classifier - probability of being "good"
            scores = self.model.predict_proba(scaled)[:, 1]
        
        return scores.flatten()
    
    def save(self, path: Union[str, Path]):
        """Save the trained model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_one_class': self.is_one_class,
                'backbone': self.backbone,
                'feature_dim': self.feature_extractor.feature_dim,
                'recommended_threshold': self.recommended_threshold,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load a trained model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        saved_backbone = data.get('backbone')
        if saved_backbone and saved_backbone != self.feature_extractor.model_name:
            raise ValueError(
                f"Preference model was trained with backbone '{saved_backbone}', "
                f"but extractor is '{self.feature_extractor.model_name}'. "
                f"Retrain with ./photoscanner.sh (Train)."
            )
        saved_dim = data.get('feature_dim')
        if saved_dim is not None and saved_dim != self.feature_extractor.feature_dim:
            raise ValueError(
                f"Preference model feature dim {saved_dim} does not match "
                f"extractor dim {self.feature_extractor.feature_dim}. Retrain."
            )
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_one_class = data.get('is_one_class', True)  # Default to one-class for old models
        self.backbone = saved_backbone or self.feature_extractor.model_name
        self.recommended_threshold = data.get('recommended_threshold')
        print(f"Model loaded from {path}")
