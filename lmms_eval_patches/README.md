# RefCOCO evaluation patches

The CatchUp LLaVA-1.5 reproduction uses the RefCOCO, RefCOCO+, and RefCOCOg evaluation utilities provided by `lmms-eval` **v0.7.1**. The three included patches preserve our inference-time bounding-box and prompt conventions.

For the released LLaVA setup, the evaluation runner sets:

- `BBOX_COORD_FORMAT=normalized`: LLaVA outputs normalized `[0,1]` bounding-box coordinates.
- `REFCOCO_PROMPT_STYLE=full`: use the original LLaVA/Shikra-style grounding instruction.

The patches also carry some upstream/research compatibility branches, but **this repository supports only LLaVA-1.5-7B**.

After installing upstream `lmms-eval` in editable mode, run from the repository root:

```bash
bash lmms_eval_patches/apply.sh
```

The helper finds the active Python installation of `lmms_eval` and skips previously patched files. If a patch cannot be applied, check that the upstream revision matches v0.7.1 rather than continuing with an unverified prompt/coordinate convention.

The target files are `lmms_eval/tasks/refcoco/utils_rec.py`, `lmms_eval/tasks/refcoco+/utils_rec.py`, and `lmms_eval/tasks/refcocog/utils_rec.py`.
