from src.core.yaml_config import YAMLConfig
from src.solver.det_solver import DetSolver

test_c = YAMLConfig("/Users/amin/Desktop/higharc/RT-DETR-plan/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_m_6x_coco_segmentation_higharc.yml")

sol = DetSolver(test_c)