import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.analysis.integrated_analysis2 import main


if __name__ == "__main__":
    main()