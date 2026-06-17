# save_webp_meta

ComfyUI node for saving images with metadata.

License: MIT

![Save WEBP Meta node](images/save-webp-meta-node.png)

## Overview

This custom node saves ComfyUI-generated images with embedded metadata so they can be viewed in prompt viewers such as SD Prompt Reader, Tiefsee, and Infinite Image Browsing.

Compatible viewers:
- [receyuki/stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader)
- [hbl917070/Tiefsee4](https://github.com/hbl917070/Tiefsee4)
- [zanllp/infinite-image-browsing](https://github.com/zanllp/infinite-image-browsing)

As a bonus, the saved images can be dragged into the prompt field of A1111-based Stable Diffusion WebUI tools such as Forge or NEO to recover part of the prompt data.
Because ComfyUI and A1111-based tools use different samplers, schedulers, and related settings, the imported result may not be exact.

[日本語↓](#日本語)

## Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ukr8b3g-cmyk/ComfyUI-save-webp-meta-node save_webp_meta
```

1. Place `ComfyUI/custom_nodes/save_webp_meta` in the right folder.
2. Restart ComfyUI.
3. Check the node list for `Save WEBP Meta` under `image/save`.

## Supported Formats

- `webp`
- `webp_lossless`
- `png`
- `jpg`
- `avif`

## Main Files

- `webp_save.py`: main node implementation
- `__init__.py`: ComfyUI node registration
- `_import_check.py`: dependency check helper
- `README.md`: documentation
- `LICENSE`: MIT license

## Default Values

- `filename_prefix`: `comfy_%model%_%date%`
- `file_format`: `webp`
- `quality`: `80`

These defaults favor ComfyUI metadata viewers and keep the saved file name compact by default.

## Filename Patterns

- `%seed%`
- `%width%`
- `%height%`
- `%pprompt:N%`
- `%nprompt:N%`
- `%model:N%`
- `%date%`
- `%date:FORMAT%`
- `%NodeTitle.WidgetName%`

## 日本語

SD prompt reader, Tiefsee, Infinite Image Browsing などのプロンプトビュアーで ComfyUI 生成画像の情報を閲覧できるようにするための、生成画像保存用カスタムノードです。

対応ビュアー:
- [receyuki/stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader)
- [hbl917070/Tiefsee4](https://github.com/hbl917070/Tiefsee4)
- [zanllp/infinite-image-browsing](https://github.com/zanllp/infinite-image-browsing)

おまけとして、Stable Diffusion WebUI の A1111 系、例えば Forge や NEO などのプロンプト欄へ D&D すると、ある程度の読み込みができます。
ComfyUI と A1111 系ではサンプラーやスケジューラなどが異なるため、正確に再現できない場合があります。

## 主要ファイル

- `webp_save.py`: ノード本体
- `__init__.py`: ComfyUI へのノード登録
- `_import_check.py`: 依存確認
- `README.md`: 説明
- `LICENSE`: MIT ライセンス

## 初期値

- `filename_prefix`: `comfy_%model%_%date%`
- `file_format`: `webp`
- `quality`: `80`

この初期値は、ComfyUI 側のメタデータ閲覧を前提にしつつ、ファイル名が長くなりすぎないようにしています。

## インストール

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ukr8b3g-cmyk/ComfyUI-save-webp-meta-node save_webp_meta
```

1. `ComfyUI/custom_nodes/save_webp_meta` に配置します。
2. ComfyUI を再起動します。
3. ノード一覧の `image/save` に `Save WEBP Meta` が表示されます。
