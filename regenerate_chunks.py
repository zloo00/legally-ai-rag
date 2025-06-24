import os
import shutil
from preprocess_articles import process_files, analyze_chunks

def cleanup_existing_chunks():
    """Remove existing chunks and regenerate with article-based approach"""
    chunk_dir = "data/chunks"
    
    print("🧹 Cleaning up existing chunks...")
    
    if os.path.exists(chunk_dir):
        # Remove all files in chunks directory
        for filename in os.listdir(chunk_dir):
            file_path = os.path.join(chunk_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"   Deleted: {filename}")
            except Exception as e:
                print(f"   Error deleting {filename}: {e}")
        
        print(f"✅ Cleaned up {chunk_dir}")
    else:
        print(f"📁 Creating {chunk_dir} directory")
        os.makedirs(chunk_dir, exist_ok=True)

def main():
    """Main function to regenerate chunks"""
    print("🔄 Regenerating chunks with article-based approach...")
    print("=" * 60)
    
    # Step 1: Clean up existing chunks
    cleanup_existing_chunks()
    
    print("\n" + "=" * 60)
    
    # Step 2: Generate new article-based chunks
    process_files()
    
    print("\n" + "=" * 60)
    
    # Step 3: Analyze the new chunks
    analyze_chunks()
    
    print("\n" + "=" * 60)
    print("✅ Chunk regeneration completed!")
    print("\n📋 Next steps:")
    print("1. Run 'python embed_and_index_openai.py' to re-index the new chunks")
    print("2. Test the RAG system with the new article-based chunks")

if __name__ == "__main__":
    main() 