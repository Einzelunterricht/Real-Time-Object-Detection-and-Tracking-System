import shutil
import logging
from pathlib import Path
from ultralytics import YOLO

# Standard Industrial Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PseudoLabelingEngine:
    """
    An automated data engineering pipeline based on the Pseudo-Labeling paradigm.
    Synchronizes class IDs across diverse datasets to maintain label consistency.
    """

    def __init__(self, model_path: str, target_class_id: int = 2):
        """
        Initializes the Perception Core.
        :param model_path: Path to the pre-trained weights (.pt or .engine).
        :param target_class_id: Unified ID for the target class (Default: 2 for self_hp).
        """
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Weight core not found at: {self.model_path}")

        # Load the SOTA Detection Backbone
        self.model = YOLO(str(self.model_path))
        self.target_id = str(target_class_id)
        logging.info(f"✅ Perception Core initialized: {self.model_path.name}")

    def execute_pipeline(self, data_dir: str, confidence: float = 0.15, imgsz: int = 640):
        """
        Executes the end-to-end labeling workflow.
        """
        target_dir = Path(data_dir).resolve()
        cache_dir = target_dir / "inference_artifacts"

        logging.info(f"🚀 Launching Pipeline | Target Domain: {target_dir}")

        # 1. Distributed Feature Inference
        # Leveraging NVIDIA GPU for high-throughput feature extraction
        self.model.predict(
            source=str(target_dir),
            conf=confidence,
            imgsz=imgsz,
            save_txt=True,
            project=str(target_dir),
            name="inference_artifacts",
            exist_ok=True,
            device=0,
            verbose=False
        )

        label_source = cache_dir / "labels"
        if not label_source.exists():
            logging.warning("⚠️ No valid targets captured. Consider adjusting the confidence threshold.")
            return

        # 2. Label Taxonomy Alignment & Normalization
        # [Image of YOLO label format conversion]
        artifact_files = list(label_source.glob("*.txt"))
        for txt_file in artifact_files:
            try:
                self._align_taxonomy(txt_file, target_dir / txt_file.name)
            except Exception as e:
                logging.error(f"Error processing {txt_file.name}: {e}")

        # 3. Resource Release & Cache Cleanup
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logging.info(f"📊 Pipeline Execution Complete | Processed: {len(artifact_files)} samples | Environment Reset.")

    def _align_taxonomy(self, src_path: Path, dst_path: Path):
        """
        Force-aligns detected class indices to a unified taxonomy.
        Crucial for resolving semantic conflicts in