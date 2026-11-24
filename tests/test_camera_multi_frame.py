import pytest
import gameplay.camera_multi_frame as mod

CASES = [
    ("frame_by_visibility", (1,)),
    ("distance_rate_limit", (1,)),
    ("occlusion_frame_tag", (1,)),
    ("pool_per_camera", (1,)),
    ("follow_authoritative", (1,)),
    ("freeze_on_rollback", (1,)),
    ("update", (1,)),
    ("occlude", (1,)),
    ("split", (1,)),
    ("present", (1,)),
    ("validate_camera", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
