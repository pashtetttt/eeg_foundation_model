# Data Layout

The repository expects local EEG datasets that are **not** stored in Git:

- `data/` for the main cohort
- `data_kids/` for external/patient cohort analyses

Expected structure is group-based (top-level class folders), with EDF files inside nested subject folders.

## Notes

- File names are used to separate open-eyes and closed-eyes recordings.
- Keep the same naming conventions used in the current archive for reproducibility.
- Large preprocessed artifacts should remain outside Git or in external storage.
