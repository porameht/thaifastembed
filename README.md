# Thai FastEmbed

A BM25-based sparse embedding library specialized for Thai text processing with advanced text processing capabilities.

## Quick Start

### Basic Usage

```python
from thaifastembed import ThaiBm25

# Initialize with default settings
bm25 = ThaiBm25()

# Sample documents
documents = [
    "กรุงเทพมหานคร เป็นเมืองหลวงของประเทศไทย",
    "กรุงเทพมหานคร มีประชากรมาก", 
    "ประเทศไทยมีวัดมากมาย"
]

# Generate embeddings
embeddings = bm25.embed(documents)
print(f"Generated {len(embeddings)} embeddings")

# Query embedding
query_embeddings = list(bm25.query_embed("กรุงเทพมหานคร"))
query_embedding = query_embeddings[0]
print(f"Query has {len(query_embedding.indices)} tokens")
```

### Custom Configuration

```python
from thaifastembed import ThaiBm25, TextProcessor, PyThaiNLPTokenizer, StopwordsFilter

# Configure text processing
tokenizer = PyThaiNLPTokenizer(engine="newmm", use_custom_dict=True)
stopwords_filter = StopwordsFilter()
text_processor = TextProcessor(
    tokenizer=tokenizer,
    lowercase=True,
    stopwords_filter=stopwords_filter,
    min_token_len=2
)

# Initialize with custom settings
bm25 = ThaiBm25(
    text_processor=text_processor,
    k=1.2,  # Term frequency saturation
    b=0.75  # Document length normalization
)
```

## Integration with Qdrant

```python
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Setup Qdrant
client = QdrantClient(url="http://localhost:6333")
collection_name = "thai_bm25"

# Create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }
)

# Index documents
documents = ["กรุงเทพมหานคร เป็นเมืองหลวงของประเทศไทย"]
embeddings = bm25.embed(documents)

points = []
for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
    points.append(PointStruct(
        id=idx,
        payload={"text": doc},
        vector={
            "text": models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist()
            )
        }
    ))

client.upsert(collection_name=collection_name, points=points)

# Search
query_embeddings = list(bm25.query_embed("กรุงเทพมหานคร"))
query_embedding = query_embeddings[0]

results = client.query_points(
    collection_name=collection_name,
    query=models.SparseVector(
        indices=query_embedding.indices.tolist(),
        values=query_embedding.values.tolist()
    ),
    using="text",
    limit=3
).points
```

## API Reference

### ThaiBm25

```python
ThaiBm25(
    text_processor: Optional[TextProcessor] = None,
    k: float = 1.2,
    b: float = 0.75,
    avg_len: float = 256.0
)

# Methods
def embed(self, documents: List[str]) -> List[SparseEmbedding]
def query_embed(self, query: Union[str, Iterable[str]]) -> Iterable[SparseEmbedding]
```

### TextProcessor

```python
TextProcessor(
    tokenizer: Optional[Tokenizer] = None,           # Default: PyThaiNLPTokenizer()
    lowercase: Optional[bool] = True,
    stopwords_filter: Optional[StopwordsFilter] = None,
    min_token_len: Optional[int] = None,
    max_token_len: Optional[int] = None
)
```

### PyThaiNLPTokenizer

Enhanced tokenizer with custom dictionary support:

```python
PyThaiNLPTokenizer(
    engine: str = "newmm",         # PyThaiNLP engine
    use_custom_dict: bool = True   # Use enhanced dictionary (words.txt + Thai words)
)
```

### SparseEmbedding

```python
class SparseEmbedding:
    indices: np.ndarray  # Token indices
    values: np.ndarray   # Token values
    
    @classmethod
    def from_dict(cls, token_dict: Dict[int, float]) -> 'SparseEmbedding'
```

## BM25 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1.2 | Term frequency saturation (1.2-2.0 recommended) |
| `b` | 0.75 | Document length normalization (0.0-1.0) |
| `avg_len` | 256.0 | Average document length for normalization |

