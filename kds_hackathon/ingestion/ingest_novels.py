#!/usr/bin/env python3
"""
Novel Ingestion Pipeline for KDS Hackathon 2026.

This module loads entire novels without truncation, chunks them intelligently
with overlapping segments, and stores them in a Pathway vector index with
timeline metadata for temporal reasoning.

Requirements:
- Full novels processed (no truncation)
- Overlapping chunks (~1000 tokens, ~20% overlap)
- Timeline awareness (normalized position 0->1)
- Pathway vector indexing
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import tiktoken
import pathway as pw
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

CHUNK_SIZE_TOKENS = 1000  # Target chunk size in tokens
OVERLAP_TOKENS = 200      # ~20% overlap
NOVELS_DIR = Path(__file__).parent.parent / "data" / "novels"
INDEX_DIR = Path(__file__).parent.parent / "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality embeddings
TOKENIZER_MODEL = "cl100k_base"  # GPT-4 tokenizer for accurate token counting


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ChunkMetadata:
    """Metadata for a single chunk of a novel."""
    story_id: str
    chunk_id: int
    text: str
    position: float  # Normalized end position (0.0 -> 1.0)
    start_token: int
    end_token: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Tokenization
# =============================================================================

class Tokenizer:
    """Token-based text processing using tiktoken."""
    
    def __init__(self, model: str = TOKENIZER_MODEL):
        self.encoding = tiktoken.get_encoding(model)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))


# =============================================================================
# Chunking Engine
# =============================================================================

class NovelChunker:
    """
    Chunks novels into overlapping segments with timeline metadata.
    
    Key properties:
    - No truncation: entire novel is processed
    - Overlapping chunks for narrative continuity
    - Position tracking for temporal reasoning
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = OVERLAP_TOKENS,
        tokenizer_model: str = TOKENIZER_MODEL
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = Tokenizer(tokenizer_model)
        self.stride = chunk_size - overlap
        
        if self.stride <= 0:
            raise ValueError("Chunk size must be greater than overlap")
    
    def chunk_novel(self, text: str, story_id: str) -> List[ChunkMetadata]:
        """
        Chunk a novel into overlapping segments with position metadata.
        
        Args:
            text: Full novel text
            story_id: Identifier for the story
            
        Returns:
            List of ChunkMetadata objects
        """
        # Tokenize entire novel
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return []
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            
            # Extract and decode chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate normalized position (end position of chunk)
            position = end_idx / total_tokens
            
            chunk = ChunkMetadata(
                story_id=story_id,
                chunk_id=chunk_id,
                text=chunk_text,
                position=position,
                start_token=start_idx,
                end_token=end_idx,
                total_tokens=total_tokens
            )
            chunks.append(chunk)
            
            chunk_id += 1
            start_idx += self.stride
            
            # Ensure we don't create tiny final chunks
            if start_idx < total_tokens and (total_tokens - start_idx) < self.overlap:
                # Extend the last chunk to include remaining tokens
                break
        
        return chunks


# =============================================================================
# Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return [emb.tolist() for emb in embeddings]


# =============================================================================
# Novel Loader
# =============================================================================

class NovelLoader:
    """Loads novels from the data directory."""
    
    def __init__(self, novels_dir: Path = NOVELS_DIR):
        self.novels_dir = novels_dir
    
    def list_novels(self) -> List[Path]:
        """List all novel files in the directory."""
        if not self.novels_dir.exists():
            raise FileNotFoundError(f"Novels directory not found: {self.novels_dir}")
        
        novels = list(self.novels_dir.glob("*.txt"))
        return sorted(novels)
    
    def load_novel(self, path: Path) -> Tuple[str, str]:
        """
        Load a novel from a file.
        
        Returns:
            Tuple of (story_id, text)
        """
        story_id = path.stem  # Filename without extension
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        return story_id, text
    
    def load_all(self) -> List[Tuple[str, str]]:
        """Load all novels from the directory."""
        novels = []
        for path in self.list_novels():
            story_id, text = self.load_novel(path)
            novels.append((story_id, text))
            print(f"Loaded: {story_id} ({len(text):,} characters)")
        return novels


# =============================================================================
# Pathway Vector Index
# =============================================================================

class PathwayVectorIndex:
    """
    Builds and manages a Pathway-based vector index for novel chunks.
    
    Uses Pathway for:
    - Data ingestion and table management
    - Vector similarity search
    - Metadata filtering
    """
    
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.index_dir / "chunks.json"
        self.embeddings_file = self.index_dir / "embeddings.pkl"
        self.index_file = self.index_dir / "index_metadata.json"
        
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._pw_table: Optional[pw.Table] = None
    
    def add_chunks(
        self, 
        chunks: List[ChunkMetadata], 
        embeddings: List[List[float]]
    ) -> None:
        """Add chunks with embeddings to the index."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data = chunk.to_dict()
            chunk_data['embedding'] = embedding
            self._chunks.append(chunk_data)
            self._embeddings.append(embedding)
    
    def build_pathway_table(self) -> pw.Table:
        """Build a Pathway table from the indexed chunks."""
        if not self._chunks:
            raise ValueError("No chunks to index")
        
        # Define Pathway schema
        class ChunkSchema(pw.Schema):
            story_id: str
            chunk_id: int
            text: str
            position: float
            start_token: int
            end_token: int
            total_tokens: int
            embedding: list
        
        # Create Pathway table from data
        # Prepare rows for Pathway ingestion
        rows = []
        for chunk in self._chunks:
            rows.append({
                'story_id': chunk['story_id'],
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'position': chunk['position'],
                'start_token': chunk['start_token'],
                'end_token': chunk['end_token'],
                'total_tokens': chunk['total_tokens'],
                'embedding': chunk['embedding']
            })
        
        # Use Pathway's debug connector for static data ingestion
        self._pw_table = pw.debug.table_from_pandas(
            __import__('pandas').DataFrame(rows)
        )
        
        return self._pw_table
    
    def save(self) -> None:
        """Persist the index to disk."""
        # Save chunks as JSON
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            # Save without embeddings in JSON for readability
            chunks_without_embeddings = [
                {k: v for k, v in chunk.items() if k != 'embedding'}
                for chunk in self._chunks
            ]
            json.dump(chunks_without_embeddings, f, indent=2)
        
        # Save embeddings separately (more efficient)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self._embeddings, f)
        
        # Save index metadata
        metadata = {
            'total_chunks': len(self._chunks),
            'total_stories': len(set(c['story_id'] for c in self._chunks)),
            'embedding_dim': len(self._embeddings[0]) if self._embeddings else 0,
            'stories': list(set(c['story_id'] for c in self._chunks))
        }
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to: {self.index_dir}")
    
    def load(self) -> 'PathwayVectorIndex':
        """Load the index from disk."""
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Index not found: {self.chunks_file}")
        
        # Load chunks
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self._chunks = json.load(f)
        
        # Load embeddings
        with open(self.embeddings_file, 'rb') as f:
            self._embeddings = pickle.load(f)
        
        # Reattach embeddings to chunks
        for chunk, embedding in zip(self._chunks, self._embeddings):
            chunk['embedding'] = embedding
        
        return self
    
    def get_chunks_by_story(self, story_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific story."""
        return [c for c in self._chunks if c['story_id'] == story_id]
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all indexed chunks."""
        return self._chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if not self._chunks:
            return {'status': 'empty'}
        
        stories = {}
        for chunk in self._chunks:
            sid = chunk['story_id']
            if sid not in stories:
                stories[sid] = {'chunk_count': 0, 'total_tokens': 0}
            stories[sid]['chunk_count'] += 1
            stories[sid]['total_tokens'] = chunk['total_tokens']
        
        return {
            'total_chunks': len(self._chunks),
            'total_stories': len(stories),
            'stories': stories,
            'embedding_dim': len(self._embeddings[0]) if self._embeddings else 0
        }


# =============================================================================
# Main Ingestion Pipeline
# =============================================================================

class IngestionPipeline:
    """
    Complete ingestion pipeline that:
    1. Loads all novels
    2. Chunks them with overlap
    3. Generates embeddings
    4. Builds Pathway vector index
    """
    
    def __init__(
        self,
        novels_dir: Path = NOVELS_DIR,
        index_dir: Path = INDEX_DIR,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = OVERLAP_TOKENS
    ):
        self.loader = NovelLoader(novels_dir)
        self.chunker = NovelChunker(chunk_size, overlap)
        self.embedder = EmbeddingGenerator()
        self.index = PathwayVectorIndex(index_dir)
    
    def run(self) -> PathwayVectorIndex:
        """Execute the full ingestion pipeline."""
        print("=" * 60)
        print("NOVEL INGESTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load all novels
        print("\n[1/4] Loading novels...")
        novels = self.loader.load_all()
        print(f"Loaded {len(novels)} novels")
        
        # Step 2: Chunk all novels
        print("\n[2/4] Chunking novels...")
        all_chunks: List[ChunkMetadata] = []
        for story_id, text in novels:
            chunks = self.chunker.chunk_novel(text, story_id)
            all_chunks.extend(chunks)
            print(f"  {story_id}: {len(chunks)} chunks")
        print(f"Total chunks: {len(all_chunks)}")
        
        # Step 3: Generate embeddings
        print("\n[3/4] Generating embeddings...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Build Pathway index
        print("\n[4/4] Building Pathway vector index...")
        self.index.add_chunks(all_chunks, embeddings)
        self.index.build_pathway_table()
        self.index.save()
        
        # Print stats
        stats = self.index.get_stats()
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total stories indexed: {stats['total_stories']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print("\nPer-story breakdown:")
        for story_id, info in stats['stories'].items():
            print(f"  {story_id}:")
            print(f"    Chunks: {info['chunk_count']}")
            print(f"    Tokens: {info['total_tokens']:,}")
        
        return self.index


# =============================================================================
# Sanity Checks
# =============================================================================

def run_sanity_checks(index: PathwayVectorIndex) -> bool:
    """
    Verify the ingestion pipeline produced valid results.
    
    Checks:
    1. Long novels produce hundreds of chunks
    2. First chunk contains story start
    3. Last chunk contains story end
    4. Position values: start near 0, end near 1, strictly increasing
    """
    print("\n" + "=" * 60)
    print("RUNNING SANITY CHECKS")
    print("=" * 60)
    
    all_passed = True
    stats = index.get_stats()
    
    for story_id in stats['stories'].keys():
        print(f"\nChecking: {story_id}")
        chunks = index.get_chunks_by_story(story_id)
        
        # Sort by chunk_id to ensure order
        chunks = sorted(chunks, key=lambda c: c['chunk_id'])
        
        # Check 1: Multiple chunks for long novels
        chunk_count = len(chunks)
        if chunk_count < 10:
            print(f"  ⚠ WARNING: Only {chunk_count} chunks (expected many more for novels)")
        else:
            print(f"  ✓ Chunk count: {chunk_count}")
        
        # Check 2: First chunk position near 0
        first_pos = chunks[0]['position']
        if first_pos > 0.1:
            print(f"  ✗ FAIL: First chunk position {first_pos:.3f} not near 0")
            all_passed = False
        else:
            print(f"  ✓ First chunk position: {first_pos:.3f}")
        
        # Check 3: Last chunk position near 1
        last_pos = chunks[-1]['position']
        if last_pos < 0.95:
            print(f"  ✗ FAIL: Last chunk position {last_pos:.3f} not near 1")
            all_passed = False
        else:
            print(f"  ✓ Last chunk position: {last_pos:.3f}")
        
        # Check 4: Positions strictly increasing
        positions = [c['position'] for c in chunks]
        is_increasing = all(positions[i] < positions[i+1] for i in range(len(positions)-1))
        if not is_increasing:
            print(f"  ✗ FAIL: Positions not strictly increasing")
            all_passed = False
        else:
            print(f"  ✓ Positions strictly increasing")
        
        # Check 5: No truncation (last chunk covers end)
        last_chunk = chunks[-1]
        if last_chunk['end_token'] != last_chunk['total_tokens']:
            print(f"  ✗ FAIL: Last chunk doesn't reach end of novel")
            all_passed = False
        else:
            print(f"  ✓ Full novel coverage (no truncation)")
        
        # Check 6: First chunk starts at beginning
        first_chunk = chunks[0]
        if first_chunk['start_token'] != 0:
            print(f"  ✗ FAIL: First chunk doesn't start at beginning")
            all_passed = False
        else:
            print(f"  ✓ First chunk starts at token 0")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SANITY CHECKS PASSED ✓")
    else:
        print("SOME SANITY CHECKS FAILED ✗")
    print("=" * 60)
    
    return all_passed


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for novel ingestion."""
    # Run the pipeline
    pipeline = IngestionPipeline()
    index = pipeline.run()
    
    # Run sanity checks
    run_sanity_checks(index)
    
    print("\nIngestion complete. Vector index built.")


if __name__ == "__main__":
    main()
