# save_webp_meta

ComfyUI node for saving images with metadata.

## Overview

This custom node saves ComfyUI-generated images with embedded metadata so they can be viewed in prompt viewers such as SD Prompt Reader, Tiefsee, and Infinite Image Browsing.

Compatible viewers:
- [receyuki/stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader)
- [hbl917070/Tiefsee4](https://github.com/hbl917070/Tiefsee4)
- [zanllp/infinite-image-browsing](https://github.com/zanllp/infinite-image-browsing)

As a bonus, the saved images can be dragged into the prompt field of A1111-based Stable Diffusion WebUI tools such as Forge or NEO to recover part of the prompt data.
Because ComfyUI and A1111-based tools use different samplers, schedulers, and related settings, the imported result may not be exact.

## Supported Formats

- `webp`
- `webp_lossless`
- `png`
- `jpg`
- `avif`

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
