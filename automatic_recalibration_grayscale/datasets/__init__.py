from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .kitti_dataset_1215_augmentation import KITTIDataset_ag
__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti_ag": KITTIDataset_ag
}
