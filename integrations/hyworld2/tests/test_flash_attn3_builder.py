from pathlib import Path

BUILDER = Path("integrations/hyworld2/build/hyworld2_flash_attn3_sm90.py")


def test_fa3_builder_is_hopper_only_and_pinned():
    source = BUILDER.read_text()
    assert 'TAG = "hyworld2-flash-attn3-py311-cu128-torch271-sm90-v1"' in source
    assert 'FLASH_ATTN_REVISION = "94345b456854a91582f6a4b26a6feda2c8e09419"' in source
    assert 'PYTHON, CUDA, TORCH = "3.11", "12.8.1", "2.7.1"' in source
    assert 'CUDA_ARCH, GPU = "9.0", "H100"' in source
    assert 'cwd=src / "hopper"' in source
    assert '"FLASH_ATTENTION_DISABLE_BACKWARD": "TRUE"' in source
    assert '"FLASH_ATTENTION_DISABLE_FP8": "TRUE"' in source
    assert '"FLASH_ATTENTION_DISABLE_FP16": "TRUE"' in source
    assert '"FLASH_ATTENTION_DISABLE_HDIM96": "TRUE"' in source
    assert '"FLASH_ATTENTION_DISABLE_HDIM192": "TRUE"' in source
    assert '"FLASH_ATTENTION_DISABLE_HDIM256": "TRUE"' in source


def test_fa3_compiles_without_renting_h100_and_smokes_on_h100():
    source = BUILDER.read_text()
    build = source[source.index("def build()") : source.index("def smoke()")]
    smoke = source[source.index("def smoke()") :]
    assert "torch.cuda.get_device_capability" not in build
    assert 'gpu=GPU' not in source[source.rfind("@app.function", 0, source.index("def build()")) : source.index("def build()")]
    assert 'gpu=GPU' in source[source.rfind("@app.function", 0, source.index("def smoke()")) : source.index("def smoke()")]
    assert 'torch.cuda.get_device_capability() != (9, 0)' in smoke


def test_fa3_smoke_matches_exact_hyworld_import_contract():
    source = BUILDER.read_text()
    smoke = source[source.index("def smoke()") :]
    assert "from flash_attn_interface import flash_attn_func" in smoke
    assert "from flash_attn_3 import flash_attn_interface as packaged_interface" in smoke
    assert 'dtype=torch.bfloat16' in smoke
    assert "torch.equal(out, out_packaged)" in smoke
    assert '"experimental": True' in source
    assert '"public_release": True' in source
