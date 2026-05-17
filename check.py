import os
from pathlib import Path

data_dir = Path("/home/mohammad/jordanian-speech-emotion-recognition/Dataset").resolve()

print(f"Data dir exists: {data_dir.exists()}")
print(f"Contents: {list(data_dir.iterdir()) if data_dir.exists() else 'N/A'}")

# Check both cases
for gender in ['male', 'female', 'Male', 'Female']:
    p = data_dir / gender
    print(f"'{gender}' exists: {p.exists()}")
    if p.exists():
        print(f"  Contents: {list(p.iterdir())[:5]}")