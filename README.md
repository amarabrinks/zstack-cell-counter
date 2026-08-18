# Z-Stack Cell Counter

A napari-based application for manual cell counting in microscopy Z-stacks. It supports four marker classes, XY downsampling, subject-aware navigation, review tracking, session recovery, quality-control flags, and structured data exports.

## Features

- Display and navigate multi-channel microscopy Z-stacks
- Preserve Z while downsampling XY for faster viewing
- Classify cells as marker 1, 2, 3, or 4
- Record cell coordinates and image metadata
- Navigate images by subject
- Resume at the first unreviewed image
- Mark images as reviewed or flag them for later review
- Automatically save counting sessions
- Export dataset-wide and per-subject results
- Detect missing, duplicate, unexpected, or unreviewed images
- Use optional JSON profiles for study-specific filename conventions

## Requirements

- Python 3.10 or newer
- napari
- NumPy
- tifffile
- QtPy
- AICSImageIO (recommended for microscopy image support)

Install the required packages with:

```bash
python -m pip install "napari[all]" numpy tifffile qtpy aicsimageio
```

## Files

- `zstack_cell_counter.py` — main application
- `mpfc_syn_profile.json` — optional metadata and quality-control profile for the accompanying mPFC SYN study

The application remains usable as a general cell counter without a study profile.

## Running the application

Keep the Python script and JSON profile in the same folder, then run:

```bash
python zstack_cell_counter.py
```

A folder picker will ask you to select the directory containing the microscopy images.

When `mpfc_syn_profile.json` is beside the script, it is loaded automatically. If it is absent, the application starts in generic mode.

You can also specify an image folder directly:

```bash
python zstack_cell_counter.py "D:\path\to\images"
```

## Basic workflow

1. Select an image folder.
2. Choose marker class 1, 2, 3, or 4.
3. Position the cursor over a cell and press **Space** to set the target.
4. Press **Enter** to add the cell.
5. Review the image and mark it complete.
6. Select **Next** or **Next unreviewed** to continue.

If you leave an unfinished image, the application asks whether to mark it complete, flag it for review, or remain on the image.

## Keyboard controls

| Key | Action |
|---|---|
| `1`–`4` | Select marker class |
| `Space` | Set target at cursor |
| `Enter` | Add the targeted cell |
| `A` | Add a cell directly at the cursor |
| `U` | Undo the most recent cell in the image |
| `[` / `]` | Move between Z planes |
| Left / Right arrow | Previous or next image |

## Output

By default, results are saved inside:

```text
<selected image folder>/cell_counter_output/
```

Important outputs include:

- `cell_counter_session.json` — resumable application state
- `annotations.csv` — complete cell-level data
- `images.csv` — image review status and cell totals
- `subject_summary.csv` — summarized results by subject, region, and layer
- `qc_flags.csv` — missing, duplicate, unexpected, flagged, or unreviewed images
- `activity_log.csv` — timestamped workflow events
- `subjects/` — convenient per-subject result files

Source microscopy images are read in place. The application does not move, rename, or modify them.

## Study profiles

Study-specific filename parsing and expected-image rules are stored separately in JSON profiles. This keeps the main application reusable across experiments.

The included mPFC profile recognizes filenames structured like:

```text
511_syn_3_pl_2.3_4.ome.tiff
```

It extracts subject, stain batch, region, cortical layer, and repetition number.

## Data privacy

Do not commit raw microscopy images, subject results, session files, or the generated `cell_counter_output` directory to this repository.

## Citation

A formal software citation and DOI will be added with the first archived release.

## License

This project is distributed under the MIT License.
