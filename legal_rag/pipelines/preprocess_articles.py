import os
import re
from typing import List, Dict, Optional
from tqdm import tqdm

RAW_DIR = "data/raw"
CHUNK_DIR = "data/chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation and Cyrillic characters
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\а-яА-Я]', '', text)
    return text.strip()

def extract_articles(text: str) -> List[Dict[str, str]]:
    """Extract articles from legal text"""
    # Pattern to match article headers in Russian/Kazakh legal documents
    # Matches: "Статья X. Title" or "Статья X" or "Article X. Title"
    article_pattern = r'(?:Статья|Article)\s+(\d+(?:-\d+)?)\.?\s*(.*?)(?=\n|$)'
    
    # Split text into articles
    articles = []
    
    # Find all article matches
    matches = list(re.finditer(article_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    if not matches:
        # If no articles found, try alternative patterns
        # Look for numbered sections
        alt_pattern = r'^(\d+)\.\s*(.*?)(?=\n\d+\.|$)'
        matches = list(re.finditer(alt_pattern, text, re.MULTILINE))
    
    if not matches:
        # If still no matches, split by chapters or sections
        chapter_pattern = r'(?:Глава|Chapter|Раздел|Section)\s+(\d+|[IVX]+)\.?\s*(.*?)(?=\n|$)'
        matches = list(re.finditer(chapter_pattern, text, re.MULTILINE | re.IGNORECASE))
    
    if not matches:
        # Last resort: split by paragraphs
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50:  # Only include substantial paragraphs
                articles.append({
                    'number': str(i + 1),
                    'title': f'Paragraph {i + 1}',
                    'content': para.strip(),
                    'type': 'paragraph'
                })
        return articles
    
    # Process matches to extract articles
    for i, match in enumerate(matches):
        article_number = match.group(1)
        article_title = match.group(2).strip() if len(match.groups()) > 1 else ""
        
        # Find the content of this article
        start_pos = match.end()
        end_pos = len(text)
        
        # Look for the next article to determine end position
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        
        # Extract article content
        article_content = text[start_pos:end_pos].strip()
        
        # Clean the content
        article_content = clean_text(article_content)
        
        # Only include articles with substantial content
        if len(article_content) > 20:
            articles.append({
                'number': article_number,
                'title': article_title,
                'content': article_content,
                'type': 'article'
            })
    
    return articles

def split_large_articles(articles: List[Dict[str, str]], max_tokens: int = 2000) -> List[Dict[str, str]]:
    """Split very large articles into smaller chunks"""
    def estimate_tokens(text: str) -> int:
        """Simple token estimation"""
        return int(len(text.split()) * 1.3)
    
    split_articles = []
    
    for article in articles:
        content = article['content']
        tokens = estimate_tokens(content)
        
        if tokens <= max_tokens:
            split_articles.append(article)
        else:
            # Split large article into paragraphs or sentences
            paragraphs = content.split('\n\n')
            
            if len(paragraphs) > 1:
                # Split by paragraphs
                current_chunk = []
                current_tokens = 0
                
                for para in paragraphs:
                    para_tokens = estimate_tokens(para)
                    
                    if current_tokens + para_tokens > max_tokens and current_chunk:
                        # Save current chunk
                        chunk_content = '\n\n'.join(current_chunk)
                        split_articles.append({
                            'number': f"{article['number']}-part{len(split_articles) + 1}",
                            'title': f"{article['title']} (Part {len(split_articles) + 1})",
                            'content': chunk_content,
                            'type': 'article_part'
                        })
                        
                        # Start new chunk
                        current_chunk = [para]
                        current_tokens = para_tokens
                    else:
                        current_chunk.append(para)
                        current_tokens += para_tokens
                
                # Add final chunk
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    split_articles.append({
                        'number': f"{article['number']}-part{len(split_articles) + 1}",
                        'title': f"{article['title']} (Part {len(split_articles) + 1})",
                        'content': chunk_content,
                        'type': 'article_part'
                    })
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = estimate_tokens(sentence)
                    
                    if current_tokens + sentence_tokens > max_tokens and current_chunk:
                        # Save current chunk
                        chunk_content = ' '.join(current_chunk)
                        split_articles.append({
                            'number': f"{article['number']}-part{len(split_articles) + 1}",
                            'title': f"{article['title']} (Part {len(split_articles) + 1})",
                            'content': chunk_content,
                            'type': 'article_part'
                        })
                        
                        # Start new chunk
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
                
                # Add final chunk
                if current_chunk:
                    chunk_content = ' '.join(current_chunk)
                    split_articles.append({
                        'number': f"{article['number']}-part{len(split_articles) + 1}",
                        'title': f"{article['title']} (Part {len(split_articles) + 1})",
                        'content': chunk_content,
                        'type': 'article_part'
                    })
    
    return split_articles

def process_files():
    """Process all files in the raw directory"""
    print("🔄 Starting article-based chunking process...")
    
    total_articles = 0
    
    for filename in tqdm(os.listdir(RAW_DIR), desc="Processing files"):
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(RAW_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Extract articles
            articles = extract_articles(text)
            
            # Split large articles if needed
            articles = split_large_articles(articles)
            
            # Save articles as chunks
            base_name = filename.replace(".txt", "")
            
            for i, article in enumerate(articles):
                # Create chunk filename
                chunk_filename = f"{base_name}_article_{article['number']}.txt"
                chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
                
                # Create chunk content with header
                chunk_content = f"Статья {article['number']}"
                if article['title']:
                    chunk_content += f". {article['title']}"
                chunk_content += f"\n\n{article['content']}"
                
                # Save chunk
                with open(chunk_path, "w", encoding="utf-8") as cf:
                    cf.write(chunk_content)
                
                # Save metadata
                metadata_filename = f"{base_name}_article_{article['number']}_meta.txt"
                metadata_path = os.path.join(CHUNK_DIR, metadata_filename)
                
                with open(metadata_path, "w", encoding="utf-8") as mf:
                    mf.write(f"article_number:{article['number']}\n")
                    mf.write(f"article_title:{article['title']}\n")
                    mf.write(f"article_type:{article['type']}\n")
                    mf.write(f"source:{filename}\n")
                    mf.write(f"content_length:{len(article['content'])}\n")
                    mf.write(f"estimated_tokens:{int(len(article['content'].split()) * 1.3)}\n")
                
                total_articles += 1
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Article-based chunking completed!")
    print(f"📊 Total articles extracted: {total_articles}")
    print(f"📁 Files processed: {len([f for f in os.listdir(RAW_DIR) if f.endswith('.txt')])}")

def analyze_chunks():
    """Analyze the created chunks"""
    print("\n📊 Analyzing chunks...")
    
    chunk_files = [f for f in os.listdir(CHUNK_DIR) if f.endswith('.txt') and not f.endswith('_meta.txt')]
    
    if not chunk_files:
        print("No chunks found!")
        return
    
    total_chunks = len(chunk_files)
    total_size = 0
    sizes = []
    
    for filename in chunk_files:
        file_path = os.path.join(CHUNK_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            size = len(content)
            sizes.append(size)
            total_size += size
    
    avg_size = total_size / total_chunks if total_chunks > 0 else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    
    print(f"📈 Chunk Statistics:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Average size: {avg_size:.0f} characters")
    print(f"   Min size: {min_size} characters")
    print(f"   Max size: {max_size} characters")
    print(f"   Total content: {total_size:,} characters")

if __name__ == "__main__":
    process_files()
    analyze_chunks() 