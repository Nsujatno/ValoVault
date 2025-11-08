import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_KEY")

# initialize supabase client
try:
    supabase: Client = create_client(url, key)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    supabase = None
