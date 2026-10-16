# Save WEBP Meta v1.2.0 — support and limitations

Baseline: ComfyUI **v0.40.0**, commit `3541246a9c997f4b8c32f9f88f3c985d02021c79`.
This is image **metadata** support, not a new model loader, GPU-generation benchmark or a guarantee of exact A1111 regeneration. Legacy v0.4.0 common node contracts remain usable; the versions are not interchangeable.

## Supported execution paths

- KSampler, KSamplerAdvanced, SamplerCustom and SamplerCustomAdvanced. Noise, guider, sampler and schedule are resolved independently; output and denoised_output are distinct.
- BasicGuider, CFGGuider, DualCFGGuider and DualModelGuider. No invented negative prompt / CFG for BasicGuider, no invented new-noise seed for DisableNoise. Embedded FLUX guidance is separate from CFG in the internal report.
- Qwen Image 2.1 (positive 0, negative 1, latent 2), Qwen Edit / Plus; Boogu and MageFlow paired conditioning; JoyImage single conditioning.
- MiniMax H3 ImageToVideo / ReferenceToVideo, AddGuide and SigmaShift, custom sampling, VAE decode and selected-frame saving. References never become the main image producer. H3 latent output 1 is not a negative prompt.
- SD/SDXL/SD3, FLUX, HiDream and Hunyuan text encoders, including different per-encoder prompts and Dual/Triple/Quadruple CLIP file identities.
- CLIPTextEncode-based Krea2, AnimaRegionalCanvas, and other models using reviewed common producer/loader paths. A generic or renamed checkpoint is recorded by its explicit filename, not classified by guessed architecture.
- Standard / rgthree LoRA paths and native LoadLoraModel / LoadLoraTextEncoder stacks. Application order, duplicate applications, negative strengths, separate MODEL/CLIP targets and sparse row numbers are retained. Disabled and zero-strength rows are excluded.
- Explicit single-image resize/crop/upscale/frame-selection paths. Multiple producer batches are assigned only when ranges are determinable; otherwise their flattened recipe is omitted. Actual IMAGE dimensions always supply Size and filename width/height. Seeds are never incremented by the saver.
- Workflow-only fallback: reviewed positional tables and preferred named widgets; uniquely mapped native subgraph boundaries have separate instance namespaces. Runtime API input is authoritative; stale UI values do not fill unresolved runtime fields.

## Pure compiler providers (optional, no new installation dependency)

The installed node must already be loaded in ComfyUI. The saver verifies its Python source and required preset data against `metadata_resolver/providers.json` before calling a closed list of pure CPU compiler methods. It does not import unknown node packages, call arbitrary FUNCTION methods, load models, run inference, access the network or change the graph.

| Provider | Pinned source |
|---|---|
| H3CharacterSheetDesigner | `ukr8b3g-cmyk/H3-Character-Sheet-Designer@9244cb6abeac2daaeabd42c984948def55f8d7ce` |
| Krea2 BBOX Canvas → Prompter → Export → Prompt Effect / Background Effect | `ukr8b3g-cmyk/Krea2-BBOX-Prompter@fd6d5edb8297d8c50ad45a8199ca523282f53832` |
| Krea2BBOXPoseGestureInjectorV1 | `ukr8b3g-cmyk/custom_nodes-krea2_bbox_pose_gesture_v1@62f0197195e5a2a12002e5d5b0eaaf6e441cd1bb` |

Older/newer commits with identical verified relevant files also match. CRLF/LF checkout differences are normalized. Modified compiler/preset files, missing providers, unidentified MiniBox classes or unresolved upstream inputs give partial metadata, never a guessed final prompt. Designer state_json and Canvas/Prompter data are not mistaken for final strings. Export output 0 is text; 1/2 are dimensions. Effect output 0 is combined text, output 1 is only the effect fragment. Pose output 0 is structured prompt data.

Updating another custom node may deliberately disable its pure-provider extraction until a new source pin is reviewed. Valid IMAGE saving, known settings and raw prompt/workflow storage continue. Custom dynamic LLM nodes and private Partner/API internals are not reconstructed.

## Formats and report

Public node ID, required widgets, hidden inputs, formats, default quality 70 and PNG/EXIF key/tag grammar are unchanged. PNG `parameters`, `prompt`, `workflow` and EXIF UserComment/ImageDescription plus the existing seven text tags are retained. The saver rejects EXIF capacity overflow instead of silently discarding a workflow; use PNG for oversized metadata. JPEG retains its existing RGB conversion. Lossless WebP now explicitly retains RGB beneath fully transparent pixels (`exact=True`), satisfying quantized RGBA pixel preservation.

An internal report is available through `SaveWebPMeta.metadata_report(...).as_dict()`. It distinguishes known, known_empty, not_applicable, unresolved and ambiguous fields. It is **not** embedded as a new metadata format. Important omissions are logged once per save. Compound conditioning is not flattened into a falsely equivalent single prompt. MODEL/CLIP LoRAs with differing strengths remain in the report/raw graph rather than being represented by an inaccurate A1111 tag. Optional LoRA hashes use unambiguous preloaded cache or exact relative-path JSON sidecars; the normal save path does not scan/read model weights.

## Verification scope

`tests/test_webp_save.py` is preserved byte-for-byte apart from platform line endings. Additional CPU tests cover source isolation, type/slot/namespace boundaries, H3 and BBOX compiler parity, dynamic stacks, image batches, and real PNG/WebP/lossless WebP/JPEG/AVIF save/read-back. `tools/validate_metadata.py` requires provider parity tests to run and rejects skips in release CI. No model or GPU is required.

Real ComfyUI frontend drag/drop, user's original metadata-free uploaded PNG provenance, interactive Windows workflows and full GPU model generation are **not** acceptance results of these CPU tests. Do not interpret the support matrix as all model weights having been loaded or benchmarked.

## 日本語

保存ノードから上流を辿る方式に変更し、無関係なSamplerのSeed・モデル・文章を拾うfallbackを廃止しました。H3静止画・Qwen 2.1・分離型Sampler・Krea2 BBOXに対応します。Krea2 BBOXは、検証済みのCanvas→Prompter→Export→Effect／Background／Poseの最終文字列を取得します。

別バージョンの未知実装や「MiniBox」という未特定ノードは、画像を保存しながらプロンプトを未解決にします。見た目やファイル名からの推測は行いません。モデル・画像生成そのものの対応追加ではなく、既存IMAGEの保存メタデータ抽出の改善です。

更新は通常のgit pull／Manager更新またはReleaseのソースZIPで行い、ComfyUIを再起動してください。Python・Pillow・NumPy・piexifの既存要件は変わりません。変更前のフォルダーやワークフローは別の場所へ保存してください。既存画像を書き換える機能はありません。

Design basis: approved internal specification v1.1 dated 2026-10-16, including the Krea2 BBOX addendum. The historical acceptance plan is not a claim that its frontend/GPU scenarios passed. Runtime implementation and verification boundaries are recorded above.
