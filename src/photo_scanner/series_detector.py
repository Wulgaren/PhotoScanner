"""
Series/burst detection for grouping similar photos.
Ensures we never suggest deleting all photos from a moment.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import imagehash
from PIL import Image
from pathlib import Path


@dataclass
class PhotoInfo:
    """Information about a single photo."""
    path: str
    uuid: str
    date: Optional[datetime]
    features: Optional[np.ndarray] = None
    score: Optional[float] = None
    phash: Optional[str] = None
    series_id: Optional[int] = None


@dataclass
class PhotoSeries:
    """A group of related photos."""
    series_id: int
    photos: List[PhotoInfo]
    best_photo_idx: Optional[int] = None
    
    @property
    def best_photo(self) -> Optional[PhotoInfo]:
        if self.best_photo_idx is not None and self.photos:
            return self.photos[self.best_photo_idx]
        return None
    
    @property
    def deletable_photos(self) -> List[PhotoInfo]:
        """Photos that can potentially be deleted (all except the best one)."""
        if len(self.photos) <= 1:
            return []  # Never delete the only photo in a series
        return [p for i, p in enumerate(self.photos) if i != self.best_photo_idx]


class SeriesDetector:
    """Detect and group photo series."""
    
    def __init__(self, 
                 time_threshold_seconds: int = 60,
                 similarity_threshold: float = 0.85,
                 phash_threshold: int = 10):
        """
        Initialize series detector.
        
        Args:
            time_threshold_seconds: Max time between photos to consider same series
            similarity_threshold: Cosine similarity threshold for embedding comparison
            phash_threshold: Hamming distance threshold for perceptual hash
        """
        self.time_threshold = timedelta(seconds=time_threshold_seconds)
        self.similarity_threshold = similarity_threshold
        self.phash_threshold = phash_threshold
    
    def compute_phash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash for an image."""
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception as e:
            print(f"Error computing phash for {image_path}: {e}")
            return None
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hex hash strings."""
        if hash1 is None or hash2 is None:
            return float('inf')
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    
    def are_related(self, photo1: PhotoInfo, photo2: PhotoInfo) -> bool:
        """Check if two photos are related (part of same series)."""
        # Check time proximity
        if photo1.date and photo2.date:
            time_diff = abs(photo1.date - photo2.date)
            if time_diff > self.time_threshold:
                return False
        
        # Check visual similarity using embeddings
        if photo1.features is not None and photo2.features is not None:
            sim = self.cosine_similarity(photo1.features, photo2.features)
            if sim >= self.similarity_threshold:
                return True
        
        # Check perceptual hash similarity
        if photo1.phash and photo2.phash:
            dist = self.hamming_distance(photo1.phash, photo2.phash)
            if dist <= self.phash_threshold:
                return True
        
        # If we only have time info and they're close, consider related
        if photo1.date and photo2.date:
            time_diff = abs(photo1.date - photo2.date)
            if time_diff <= timedelta(seconds=5):  # Very close in time
                return True
        
        return False
    
    def detect_series(self, photos: List[PhotoInfo]) -> List[PhotoSeries]:
        """
        Detect photo series using Union-Find algorithm.
        
        Args:
            photos: List of PhotoInfo objects with features and dates
            
        Returns:
            List of PhotoSeries objects
        """
        if not photos:
            return []
        
        n = len(photos)
        
        # Sort by date for efficient comparison
        sorted_indices = sorted(range(n), 
                               key=lambda i: photos[i].date or datetime.min)
        
        # Union-Find data structure
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        # Compare photos within time window
        print("Detecting photo series...")
        for i, idx1 in enumerate(sorted_indices):
            photo1 = photos[idx1]
            
            # Only compare with nearby photos (within time window)
            for j in range(i + 1, min(i + 100, n)):  # Look at next 100 photos max
                idx2 = sorted_indices[j]
                photo2 = photos[idx2]
                
                # Early termination if too far apart in time
                if photo1.date and photo2.date:
                    if photo2.date - photo1.date > self.time_threshold * 2:
                        break
                
                if self.are_related(photo1, photo2):
                    union(idx1, idx2)
        
        # Group photos by their root
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(i)
        
        # Create PhotoSeries objects
        series_list = []
        for series_id, indices in enumerate(groups.values()):
            series_photos = [photos[i] for i in indices]
            
            # Assign series_id to each photo
            for photo in series_photos:
                photo.series_id = series_id
            
            # Sort by date within series
            series_photos.sort(key=lambda p: p.date or datetime.min)
            
            series = PhotoSeries(series_id=series_id, photos=series_photos)
            series_list.append(series)
        
        print(f"Found {len(series_list)} photo series from {n} photos")
        return series_list
    
    def rank_series(self, series_list: List[PhotoSeries]) -> List[PhotoSeries]:
        """
        Rank photos within each series and identify the best one.
        
        Args:
            series_list: List of PhotoSeries with scored photos
            
        Returns:
            Updated series list with best_photo_idx set
        """
        for series in series_list:
            if not series.photos:
                continue
            
            # Find the photo with highest score
            scores = [p.score if p.score is not None else 0.5 for p in series.photos]
            series.best_photo_idx = int(np.argmax(scores))
        
        return series_list


def get_deletion_suggestions(series_list: List[PhotoSeries], 
                            score_threshold: float = 0.3) -> List[PhotoInfo]:
    """
    Get photos that can be safely deleted.
    
    Rules:
    1. Never delete the only photo in a series
    2. Never delete the best photo in a series
    3. Only suggest photos below the score threshold
    
    Args:
        series_list: Ranked photo series
        score_threshold: Only suggest deletion for photos below this score
        
    Returns:
        List of photos that can be deleted
    """
    deletable = []
    
    for series in series_list:
        if len(series.photos) <= 1:
            continue  # Keep sole photo
        
        for i, photo in enumerate(series.photos):
            if i == series.best_photo_idx:
                continue  # Keep the best
            
            if photo.score is not None and photo.score < score_threshold:
                deletable.append(photo)
    
    # Sort by score (lowest first = most confident deletions)
    deletable.sort(key=lambda p: p.score or 0)
    
    return deletable
