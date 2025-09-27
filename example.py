#!/usr/bin/env python3
"""
Example usage of Thai FastEmbed library
"""

from thaifastembed import ThaiBm25, SparseEmbedding, Tokenizer, TextProcessor, StopwordsFilter

def main():
    # Sample Thai documents
    documents = [
        "ประเทศไทยมีวัฒนธรรมที่หลากหลาย",
        "อาหารไทยมีรสชาติเผ็ด หวาน เปรียว เค็ม",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "ภาษาไทยเป็นภาษาราชการ",
        "การท่องเที่ยวในประเทศไทยมีความสำคัญต่อเศรษฐกิจ"
    ]
    
    print("🇹🇭 Thai FastEmbed Example")
    print("=" * 50)
    
    # 1. Basic tokenization
    print("\n1. Tokenization Example:")
    tokenizer = Tokenizer()
    text = "ประเทศไทยมีวัฒนธรรมที่หลากหลาย"
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    
    # 2. Text processing with stopwords filtering
    print("\n2. Text Processing with Stopwords:")
    stopwords_filter = StopwordsFilter()
    processor = TextProcessor(tokenizer, lowercase=True, stopwords_filter=stopwords_filter, min_token_len=1)
    
    processed_tokens = processor.process_text(text)
    filtered_tokens = [token for token in tokens if not stopwords_filter.is_stopword(token)]
    print(f"Processed tokens: {processed_tokens}")
    print(f"Manual filtered: {filtered_tokens}")
    print(f"Stopwords count: {stopwords_filter.len()}")
    
    # 3. BM25 embeddings
    print("\n3. BM25 Embeddings:")
    bm25 = ThaiBm25(text_processor=processor)
    
    # Embed documents
    doc_embeddings = bm25.embed(documents)
    print(f"Created {len(doc_embeddings)} document embeddings")
    
    # Query example
    query = "วัฒนธรรมไทย"
    query_embedding = bm25.query_embed(query)
    
    print(f"Query: {query}")
    print(f"Query embedding token count: {len(query_embedding.indices)}")
    print(f"Query tokens and IDs:")
    query_tokens = processor.process_text(query)
    for token in query_tokens:
        token_id = ThaiBm25.compute_token_id(token)
        print(f"  '{token}' -> {token_id}")
    
    # 4. Document embedding details
    print("\n4. Document Embedding Details:")
    for i, embedding in enumerate(doc_embeddings[:2]):  # Show first 2
        print(f"Document {i+1}: '{documents[i]}'")
        print(f"  Tokens: {len(embedding.indices)}")
        print(f"  Sample indices: {embedding.indices[:5] if len(embedding.indices) > 5 else embedding.indices}")
        print(f"  Sample values: {embedding.values[:5] if len(embedding.values) > 5 else embedding.values}")
    
    print(f"\n✅ Successfully processed {len(documents)} documents with Thai BM25!")

if __name__ == "__main__":
    main()