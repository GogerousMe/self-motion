"""Direct project entry point; the same model/environment runs V1, MT and MSTd."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from mstd.cli import main


if __name__=='__main__':
    main(['simulate',*sys.argv[1:]])
