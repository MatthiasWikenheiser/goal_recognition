import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("PATH_ERROR"))
os.listdir(os.getenv("PATH_ERROR"))