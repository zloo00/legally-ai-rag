import os
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔍 Testing OpenAI API Connection...")
    print(f"API Key present: {'Yes' if OPENAI_API_KEY else 'No'}")
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Test with a simple text
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=["Test text for embedding"],
            model="text-embedding-3-small"
        )
        
        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            print(f"✅ OpenAI API working! Embedding dimension: {len(embedding)}")
            return True
        else:
            print("❌ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_local_embeddings():
    """Test local sentence transformer embeddings"""
    print("\n🔍 Testing Local Embeddings...")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        text = "Test text for local embedding"
        embedding = model.encode(text)
        
        print(f"✅ Local embeddings working! Embedding dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Local embedding error: {e}")
        return False

def test_sample_chunks():
    """Test embedding generation on sample chunks"""
    print("\n🔍 Testing Sample Chunks...")
    
    chunk_dir = "data/chunks"
    sample_files = []
    
    # Get first 3 chunk files
    for filename in os.listdir(chunk_dir):
        if filename.endswith(".txt") and not filename.endswith("_meta.txt"):
            sample_files.append(filename)
            if len(sample_files) >= 3:
                break
    
    if not sample_files:
        print("❌ No chunk files found!")
        return
    
    print(f"📋 Testing {len(sample_files)} sample chunks...")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    for i, filename in enumerate(sample_files, 1):
        file_path = os.path.join(chunk_dir, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            
            print(f"\n{i}. {filename}")
            print(f"   Text length: {len(text)} characters")
            print(f"   Text preview: {text[:100]}...")
            
            # Test OpenAI embedding
            try:
                response = client.embeddings.create(
                    input=[text],
                    model="text-embedding-3-small"
                )
                
                if response.data and len(response.data) > 0:
                    embedding = response.data[0].embedding
                    print(f"   ✅ OpenAI embedding: {len(embedding)} dimensions")
                else:
                    print(f"   ❌ OpenAI embedding failed: empty response")
                    
            except Exception as e:
                print(f"   ❌ OpenAI embedding error: {e}")
            
            # Test local embedding
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                local_embedding = model.encode(text)
                print(f"   ✅ Local embedding: {len(local_embedding)} dimensions")
                
            except Exception as e:
                print(f"   ❌ Local embedding error: {e}")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")

def check_environment():
    """Check environment setup"""
    print("🔍 Checking Environment Setup...")
    
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"PINECONE_API_KEY: {'Set' if os.getenv('PINECONE_API_KEY') else 'Not set'}")
    print(f"PINECONE_ENVIRONMENT: {os.getenv('PINECONE_ENVIRONMENT', 'Not set')}")
    print(f"PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME', 'Not set')}")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file exists")
    else:
        print("❌ .env file not found")

def main():
    """Main debug function"""
    print("🐛 Debugging Embedding Issues")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Test OpenAI connection
    openai_working = test_openai_connection()
    
    # Test local embeddings
    local_working = test_local_embeddings()
    
    # Test sample chunks
    if openai_working or local_working:
        test_sample_chunks()
    
    print("\n" + "=" * 50)
    print("🔧 Debug Summary:")
    print(f"   OpenAI API: {'✅ Working' if openai_working else '❌ Failed'}")
    print(f"   Local Embeddings: {'✅ Working' if local_working else '❌ Failed'}")
    
    if not openai_working:
        print("\n💡 OpenAI API Issues - Possible Solutions:")
        print("1. Check your OPENAI_API_KEY in .env file")
        print("2. Verify your OpenAI account has credits")
        print("3. Check if the API key is valid")
        print("4. Try using local embeddings as fallback")

if __name__ == "__main__":
    main() 