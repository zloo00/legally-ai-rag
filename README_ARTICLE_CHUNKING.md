# Article-Based Chunking for Legal Documents

This project now uses **article-based chunking** instead of arbitrary character-based chunking for legal documents. This approach provides much better semantic coherence and retrieval accuracy.

## 🎯 Why Article-Based Chunking?

### Problems with Symbol-Based Chunking:
- ❌ Breaks articles in the middle of sentences
- ❌ Loses legal context and structure
- ❌ Makes it hard to find specific articles
- ❌ Poor retrieval accuracy for legal queries

### Benefits of Article-Based Chunking:
- ✅ Each chunk represents a complete legal article
- ✅ Preserves legal structure and context
- ✅ Easy to reference specific articles
- ✅ Better semantic coherence
- ✅ Improved retrieval accuracy

## 📊 Results

The new chunking approach extracted:
- **872 total articles** from 5 legal documents
- **744 final chunks** (some large articles were split into parts)
- **Average chunk size**: 1,126 characters
- **Total content**: 837,425 characters

## 🔧 How It Works

### 1. Article Detection
The system uses regex patterns to identify article boundaries:
```python
# Primary pattern for Russian/Kazakh legal documents
article_pattern = r'(?:Статья|Article)\s+(\d+(?:-\d+)?)\.?\s*(.*?)(?=\n|$)'
```

### 2. Fallback Patterns
If no articles are found, the system tries:
- Numbered sections: `^(\d+)\.\s*(.*?)(?=\n\d+\.|$)`
- Chapters/Sections: `(?:Глава|Chapter|Раздел|Section)\s+(\d+|[IVX]+)`
- Paragraphs: Split by double newlines

### 3. Large Article Handling
Articles longer than 2,000 tokens are split into parts:
- **By paragraphs** (preferred)
- **By sentences** (fallback)

### 4. Metadata Generation
Each chunk includes rich metadata:
```
article_number:1
article_title:Отношения, регулируемые гражданским законодательством
article_type:article
source:civil_code_kz.txt
content_length:1267
estimated_tokens:180
```

## 📁 File Structure

### Input Files
```
data/raw/
├── civil_code_kz.txt
├── constitution_kz.txt
├── labor_code_kz.txt
└── ...
```

### Output Chunks
```
data/chunks/
├── civil_code_kz_article_1.txt
├── civil_code_kz_article_1_meta.txt
├── civil_code_kz_article_2.txt
├── civil_code_kz_article_2_meta.txt
└── ...
```

## 🚀 Usage

### Generate New Article-Based Chunks
```bash
# Generate article-based chunks
python preprocess_articles.py

# Or regenerate all chunks (cleans up old ones first)
python regenerate_chunks.py
```

### Re-index with New Chunks
```bash
# Re-index the new article-based chunks
python embed_and_index_openai.py
```

## 📈 Chunk Examples

### Before (Symbol-Based)
```
Статья 1. Отношения, регулируемые гражданским законодательством

1. Гражданским законодательством регулируются товарно-денежные и иные основанные на равенстве участников имущественные отношения, а также связанные с имущественными личные неимущественные отношения. Участниками регулируемых гражданским законодательством отношений являются граждане, юридические лица, государство, а также административно-территориальные единицы.

2. Личные неимущественные отношения, не связанные с имущественными, регулируются гражданским законодательством, поскольку иное не предусмотрено законодательными актами либо не вытекает из существа личного неимущественного отношения.

3. К семейным, трудовым отношениям и отношениям по использованию природных ресурсов и охране окружающей среды, отвечающим признакам, указанным в пункте 1 настоящей статьи, гражданское законодательство применяется в случаях, когда эти отношения не регулируются соответственно семейным, трудовым законодательством, законодательством об использовании природных ресурсов и охране окружающей среды.

4. К имущественным отношениям, основанным на административном или ином властном подчинении одной стороны другой, в том числе к налоговым и другим бюджетным отношениям, гражданское законодательство не применяется, за исключением случаев, предусмотренных законодательными актами.
```

### After (Article-Based)
Each article is now a complete, coherent chunk with:
- Clear article header
- Complete legal content
- Proper metadata
- Semantic coherence

## 🔍 Benefits for RAG System

### 1. Better Query Matching
- Queries about specific articles return the complete article
- Legal citations are preserved
- Context is maintained

### 2. Improved Accuracy
- No broken sentences or incomplete legal concepts
- Better semantic understanding
- More relevant search results

### 3. Enhanced User Experience
- Users can reference specific articles
- Clear source attribution
- Better legal advice quality

## 🛠️ Customization

### Adjusting Article Detection
Modify patterns in `preprocess_articles.py`:
```python
# For different legal document formats
article_pattern = r'(?:Статья|Article|§|Section)\s+(\d+(?:-\d+)?)\.?\s*(.*?)(?=\n|$)'
```

### Changing Chunk Size Limits
```python
# In split_large_articles function
max_tokens = 2000  # Adjust as needed
```

### Adding New Document Types
Add new patterns for different legal systems:
```python
# For US legal documents
us_pattern = r'§\s*(\d+(?:-\d+)?)\.?\s*(.*?)(?=\n|$)'
```

## 📋 Migration Guide

### From Old Chunks to New
1. **Backup existing chunks** (optional)
2. **Run regeneration script**:
   ```bash
   python regenerate_chunks.py
   ```
3. **Re-index the new chunks**:
   ```bash
   python embed_and_index_openai.py
   ```
4. **Test the RAG system** with legal queries

### Testing the New System
```python
from rag_system import rag_system

# Test with article-specific queries
response = rag_system.query("What does Article 1 of the Civil Code say about property relations?")
print(response["answer"])
```

## 🎯 Best Practices

1. **Always use article-based chunking** for legal documents
2. **Keep articles intact** unless they're extremely long
3. **Preserve legal structure** and numbering
4. **Include rich metadata** for better retrieval
5. **Test with legal queries** to ensure quality

## 🔮 Future Improvements

- [ ] Support for more legal document formats
- [ ] Automatic article numbering detection
- [ ] Cross-reference linking between articles
- [ ] Hierarchical chunking (articles → paragraphs → sentences)
- [ ] Legal citation extraction and linking 