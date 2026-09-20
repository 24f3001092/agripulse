import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'catalog'))
