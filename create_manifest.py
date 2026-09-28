import gdown
import json
import os

fake_url = 'https://drive.google.com/drive/folders/13OqZH_uwD9IWhoWF5h0l-dTKFyRfII7A'
real_url = 'https://drive.google.com/drive/folders/1bQahWnXPid84b7MjE-9_mPyHHhzfQXH3'

print('[*] Querying Fake Images Folder from Google Drive...')
fake_files = gdown.download_folder(fake_url, skip_download=True, quiet=True)
print(f'[*] Found {len(fake_files)} Fake Image IDs.')

print('[*] Querying Real Images Folder from Google Drive...')
real_files = gdown.download_folder(real_url, skip_download=True, quiet=True)
print(f'[*] Found {len(real_files)} Real Image IDs.')

manifest = {
    'fake': [{'id': f.id, 'name': f.path} for f in fake_files],
    'real': [{'id': f.id, 'name': f.path} for f in real_files]
}

os.makedirs('data', exist_ok=True)
with open('data/manifest.json', 'w', encoding='utf-8') as fp:
    json.dump(manifest, fp, indent=2)

print(f"[OK] Successfully saved manifest.json with {len(manifest['real'])} Real and {len(manifest['fake'])} Fake images.")
