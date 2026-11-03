from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))

from prototype_patch.class_guided_cli import main


if __name__ == "__main__":
    main()
