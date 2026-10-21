# Dataset manifests

No real-data result in this repository is produced without a manifest. The
forensic runners (`experiments/forensics/real_media_null.py`, `heldout_detection.py`,
`compression_sweep.py`) exit with an error when one is missing; they
never substitute synthetic data.

## Format

CSV, one row per media file (`forensics/io/manifests.py`):

| column      | meaning |
|-------------|---------|
| `path`      | file path, relative to the repository root (or absolute) |
| `label`     | `real` or `generated` |
| `source`    | scene / prompt / source-video identifier; all rows sharing a source must be in one split |
| `generator` | generator name, `none` for real media |
| `split`     | `train_null` (fit the empirical null), `val` (the real-media null gate, thresholds), `cal` (fusion and calibration), `test` (held-out generators only) |
| `sha256`    | file checksum; verified every time the manifest is loaded |

`manifest_template.csv` shows the layout. The rows in it are examples and do not point at real files.

## Building one

Arrange the files as

    data/images/real/<source>/<file>
    data/images/generated/<generator>/<source>/<file>

then run

    python scripts/build_manifest.py --root data/images --out data/manifests/images.csv \
        --held-out-generators <generators reserved for test>

Authentic sources are hashed into the four splits. Generated items from held-out generators go to `test`; all other generated items go to `cal`.

## Pre-registration

Fix `configs/default.yaml` (the gate thresholds in particular) and commit it
**before** running `heldout_detection.py` on the test split. The config hash and
commit are written into every output, so later changes are visible.

## Where the data comes from

See `DATA.md` at the repository root for the candidate datasets and their licences. Datasets are not redistributed here.
