'''Reusable script to create your deeplake datastore

To be run after creating .env, before github.py'''

import os 
import deeplake 
from dotenv import load_dotenv

load_dotenv()
activeloop_token = os.getenv('ACTIVELOOP_TOKEN')
deeplake_username = os.getenv('DEEPLAKE_USERNAME')
deeplake_repo_name = os.getenv('DEEPLAKE_REPO_NAME')

# Overwrite = True to overwrite existing datastore & reuse if needed
ds = deeplake.empty(f'hub://{deeplake_username}/{deeplake_repo_name}',
                    token=activeloop_token,
                    overwrite=True)

# Mimic structure that github.py will create
ds.create_tensor("ids")
ds.create_tensor("metadata")
ds.create_tensor("embedding")
ds.create_tensor("text")
