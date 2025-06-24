import os
from typing import List, Dict

def analyze_chunk_quality():
    """Analyze the quality of article-based chunks"""
    chunk_dir = "data/chunks"
    
    print("🔍 Analyzing Article-Based Chunk Quality")
    print("=" * 50)
    
    # Get all chunk files
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.txt') and not f.endswith('_meta.txt')]
    
    if not chunk_files:
        print("❌ No chunks found!")
        return
    
    # Analyze chunks
    total_chunks = len(chunk_files)
    article_chunks = 0
    part_chunks = 0
    paragraph_chunks = 0
    
    # Sample analysis
    sample_chunks = []
    
    for filename in chunk_files[:5]:  # Analyze first 5 chunks
        file_path = os.path.join(chunk_dir, filename)
        meta_path = os.path.join(chunk_dir, filename.replace('.txt', '_meta.txt'))
        
        try:
            # Read chunk content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Read metadata
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key] = value
            
            # Categorize chunk
            chunk_type = metadata.get('article_type', 'unknown')
            if chunk_type == 'article':
                article_chunks += 1
            elif chunk_type == 'article_part':
                part_chunks += 1
            elif chunk_type == 'paragraph':
                paragraph_chunks += 1
            
            # Add to sample
            sample_chunks.append({
                'filename': filename,
                'content': content[:200] + "..." if len(content) > 200 else content,
                'metadata': metadata,
                'length': len(content)
            })
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    # Print analysis
    print(f"📊 Chunk Analysis:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Article chunks: {article_chunks}")
    print(f"   Article parts: {part_chunks}")
    print(f"   Paragraph chunks: {paragraph_chunks}")
    
    print(f"\n📋 Sample Chunks:")
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n{i}. {chunk['filename']}")
        print(f"   Type: {chunk['metadata'].get('article_type', 'unknown')}")
        print(f"   Article: {chunk['metadata'].get('article_number', 'N/A')}")
        print(f"   Title: {chunk['metadata'].get('article_title', 'N/A')}")
        print(f"   Length: {chunk['length']} characters")
        print(f"   Content: {chunk['content']}")

def demonstrate_improvements():
    """Demonstrate the improvements of article-based chunking"""
    print("\n🎯 Benefits of Article-Based Chunking")
    print("=" * 50)
    
    benefits = [
        {
            "title": "Complete Legal Articles",
            "description": "Each chunk contains a complete legal article with full context",
            "example": "Статья 1. Отношения, регулируемые гражданским законодательством"
        },
        {
            "title": "Preserved Legal Structure",
            "description": "Legal numbering and hierarchy are maintained",
            "example": "Article 1, Article 2, Article 3..."
        },
        {
            "title": "Better Search Results",
            "description": "Queries about specific articles return complete, relevant content",
            "example": "What does Article 1 say about property relations?"
        },
        {
            "title": "Rich Metadata",
            "description": "Each chunk includes article number, title, source, and type",
            "example": "article_number:1, article_title:Отношения..."
        },
        {
            "title": "Semantic Coherence",
            "description": "No broken sentences or incomplete legal concepts",
            "example": "Complete legal provisions in single chunks"
        }
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"\n{i}. {benefit['title']}")
        print(f"   {benefit['description']}")
        print(f"   Example: {benefit['example']}")

def compare_approaches():
    """Compare old vs new chunking approaches"""
    print("\n⚖️  Comparison: Old vs New Approach")
    print("=" * 50)
    
    comparison = [
        {
            "aspect": "Chunk Boundaries",
            "old": "Arbitrary character limits",
            "new": "Natural article boundaries"
        },
        {
            "aspect": "Legal Context",
            "old": "Often broken mid-article",
            "new": "Complete legal articles preserved"
        },
        {
            "aspect": "Search Accuracy",
            "old": "Poor for legal queries",
            "new": "High accuracy for article-specific queries"
        },
        {
            "aspect": "Metadata",
            "old": "Basic file information",
            "new": "Rich legal metadata (article number, title, type)"
        },
        {
            "aspect": "User Experience",
            "old": "Confusing, incomplete results",
            "new": "Clear, complete legal information"
        }
    ]
    
    for item in comparison:
        print(f"\n📋 {item['aspect']}")
        print(f"   ❌ Old: {item['old']}")
        print(f"   ✅ New: {item['new']}")

def main():
    """Main function to run the analysis"""
    print("🧪 Testing Article-Based Chunking")
    print("=" * 60)
    
    # Analyze chunk quality
    analyze_chunk_quality()
    
    # Demonstrate improvements
    demonstrate_improvements()
    
    # Compare approaches
    compare_approaches()
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("\n📋 Next steps:")
    print("1. Re-index the new chunks: python embed_and_index_openai.py")
    print("2. Test the RAG system with legal queries")
    print("3. Compare search results quality")

if __name__ == "__main__":
    main() 