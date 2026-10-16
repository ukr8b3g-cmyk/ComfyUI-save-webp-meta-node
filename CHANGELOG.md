# Changelog

## 1.2.0 — 2026-10-17 (JST)

- Follow only the image connected to the saving node; never fall back to an unrelated first sampler or stale workflow value.
- Add H3 static-frame, custom sampling, paired edit-conditioning, multi-text/encoder and native MODEL/CLIP LoRA-stack metadata paths.
- Add optional source-pinned pure compiler providers for H3 Character Sheet Designer and Krea2 BBOX Canvas/Prompter/Export/Effect/Background/Pose.
- Preserve exact 64-bit seeds, known-empty negative prompts, output slots, source namespaces, multi-stage identity and provable batch ranges. Use actual output pixel dimensions.
- Keep existing public I/O, image formats and raw graph metadata grammar. Reject metadata overflow rather than truncating it; retain transparent RGB in lossless WebP.
- Add read-only internal resolution reports, bounded per-save memoization, source/codec regressions and Linux/Windows CPU release gates.
- No new mandatory runtime dependency; no sampling/GPU execution in the saver. Existing Release and Registry publication workflows remain unchanged.

Known limits: unverified provider versions and dynamic/unknown nodes yield partial metadata; frontend workflow restoration and actual GPU generation are separate acceptance work.
