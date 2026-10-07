"""Static contract checks for the production full-pipeline launcher."""

from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / 'run_full_pipeline.sh'


def test_launcher_keeps_full_default_and_propagates_one_profile_to_every_stage():
    text = LAUNCHER.read_text()
    assert 'SOURCE_SELECTION_PROFILE="${6:-activitynet_full_v1}"' in text
    assert '[[ $# -ge 3 && $# -le 6 ]]' in text
    assert text.count('"source_selection=$SOURCE_SELECTION_PROFILE"') == 6
    for module in (
        'scripts.moco.train_full_streaming_moco',
        'scripts.linear_probe.build_manifest',
        'scripts.linear_probe.extract_features',
        'scripts.linear_probe.run_probe',
    ):
        assert module in text


def test_launcher_validates_exact_snapshot_selection_before_manifest_stage():
    text = LAUNCHER.read_text()
    verification = text.index('resolve_snapshot_manifest_id')
    manifest_stage = text.index('# 2. ActivityNet segment manifest')
    assert verification < manifest_stage
    assert '"$DATASET_ROOT" "$FINAL_SNAPSHOT/metadata.json" "$SOURCE_SELECTION_PROFILE"' in text
    subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)
