
# First, add the directory containing your custom modules to the Python path
import sys
sys.path.append('/home/work/Project/Lisa/model/llava/model/multimodal_encoder')

# Now you can import your custom module
from clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if (
        vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "clip" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
