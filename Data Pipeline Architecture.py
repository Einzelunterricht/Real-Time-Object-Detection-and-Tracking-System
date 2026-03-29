import random
import shutil
import logging
from pathlib import Path

# Industrial-grade logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class DatasetStratifier:
    """
    Automates the split of raw data into training and validation subsets.
    Ensures structural integrity by verifying the correlation between images and labels.
    """

    def __init__(self, source_dir: str, output_base: str, split_ratio: float = 0.8):
        self.source_dir = Path(source_dir).resolve()
        self.output_base = Path(output_base).resolve()
        self.split_ratio = split_ratio

        # Define the standard YOLO directory topology
        self.splits = ['train', 'val']
        self.subdirs = ['images', 'labels']

    def initialize_structure(self):
        """Creates the directory hierarchy required for YOLO training."""
        for split in self.splits:
            for subdir in self.subdirs:
                (self.output_base / subdir / split).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory topology initialized at: {self.output_base}")

    def execute_stratification(self):
        """Performs randomized data splitting and file relocation."""
        # Step 1: Filter valid samples (Intersection of JPG and TXT sets)
        all_images = {f.stem for f in self.source_dir.glob("*.jpg")}
        valid_samples = [stem for stem in all_images if (self.source_dir / f"{stem}.txt").exists()]

        if not valid_samples:
            logging.error("No valid annotated samples found. Aborting.")
            return

        logging.info(f"Detected valid annotated samples: {len(valid_samples)}")

        # Step 2: Randomized shuffling for statistical independence
        random.shuffle(valid_samples)

        # Step 3: Compute split index
        threshold = int(len(valid_samples) * self.split_ratio)
        distribution = {
            'train': valid_samples[:threshold],
            'val': valid_samples[threshold:]
        }

        # Step 4: Asset relocation
        for split, file_list in distribution.items():
            self._relocate_assets(file_list, split)
            logging.info(f"Split [{split}]: Successfully allocated {len(file_list)} samples.")

    def _relocate_assets(self, file_list, split):
        """Internal helper to copy high-resolution imagery and ground-truth labels."""
        for stem in file_list:
            # Source paths
            img_src = self.source_dir / f"{stem}.jpg"
            lbl_src = self.source_dir / f"{stem}.txt"

            # Destination paths (Standard YOLO format)
            shutil.copy(img_src, self.output_base / 'images' / split / f"{stem}.jpg")
            shutil.copy(lbl_src, self.output_base / 'labels' / split / f"{stem}.txt")


if __name__ == "__main__":
    # Project Root: D:\04 Project\Object-Tracking-System
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # Configuration Matrix
    CONFIG = {
        "src": PROJECT_ROOT / "raw_data",  # Your source pool of images/labels
        "dst": PROJECT_ROOT / "dataset",  # Final structured dataset for YOLO
        "ratio": 0.8
    }

    # Pipeline Execution
    stratifier = DatasetStratifier(CONFIG["src"], CONFIG["dst"], CONFIG["ratio"])
    stratifier.initialize_structure()
    stratifier.execute_stratification()