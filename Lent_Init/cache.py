import hashlib
import pickle
from pathlib import Path
import os
#caches the responses from the api so I don't have to keep calling with the same prompt when testing other things
class Cache:
    def __init__(self, cache_dir: str = "cache"):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the cache directory
        self.cache_dir = Path(current_dir) / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")
        
    def clear(self):
        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
            except Exception as e:
                print(f"ERROR deleting {file}: {e}")
        print("Cache cleared")
    
    def clear_invalid(self):
        for file in self.cache_dir.glob("*.pkl"):
            try:
                with open(file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Check for empty or invalid responses
                if isinstance(cached_data, list) and not cached_data:
                    file.unlink()
                elif isinstance(cached_data, str) and not cached_data.strip():
                    file.unlink()
                    
            except Exception as e:
                print(f"ERROR checking {file}: {e}")
                file.unlink()  # Delete files that can't be loaded
        print("Invalid entries cleared")
        
    # makes a unique hash key to store then get each abstract and generated summary
    def make_hash_key(self, abstract: str, gen_summary: str) -> str:
        # Ensure both strings are properly encoded before hashing
        if abstract is None:
            abstract = ""
        if gen_summary is None:
            gen_summary = ""
            
        # Convert to string if not already
        abstract_str = str(abstract)
        gen_summary_str = str(gen_summary)
        
        # Create the combined string and encode it
        combined = f"{gen_summary_str}:{abstract_str}"
        encoded = combined.encode('utf-8', errors='replace')
        
        return hashlib.md5(encoded).hexdigest()
        
    def get(self, abstract: str, gen_summary: str):
        # uses the abstract and gen summary to make a unique hash
        cache_key = self.make_hash_key(abstract, gen_summary)
        # sets the dir for the cache_file to be in the generated unique hash .pkl file
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # if that .pkl file exists, then that means there exists the exact same abstract in the cache
        if cache_file.exists():
            try:
                #loads the cache data 
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # should return either summary or triplets depending on what we are pulling
                print(f"Cache hit for {gen_summary}")
                return cached_data
            except Exception as e:
                print(f"Cache read error: {e}")
                return None
        return None
        
    def set(self, abstract: str, gen_summary: str, result):
        # does the same process as the getting, but instead creates the file
        cache_key = self.make_hash_key(abstract, gen_summary)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Opens the file name it made in writing binary mode so we can pickle dump
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Cache write error: {e}")

class SimpleCache:
    """Simplified cache for refinement task."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Simple cache dir: {self.cache_dir}")

    def get(self, key_text: str):
        cache_key = hashlib.md5(key_text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, key_text: str, result):
        cache_key = hashlib.md5(key_text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Write failed: {e}")
