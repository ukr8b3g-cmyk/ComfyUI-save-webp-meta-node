"""Explicit node/slot contracts. Baseline: ComfyUI v0.40.0, see docs/SOURCES.json.

An unknown node is NOT made transparent because its name or input resembles a
known node. These tables describe provenance, not model inference support.
"""
SAMPLERS = {"KSampler": 1, "KSamplerAdvanced": 1, "SamplerCustom": 2, "SamplerCustomAdvanced": 2}
TEXT_ENCODERS = {
    "CLIPTextEncode": ("text",),
    "CLIPTextEncodeSDXL": ("text_g", "text_l"),
    "CLIPTextEncodeSDXLRefiner": ("text",),
    "CLIPTextEncodeSD3": ("clip_l", "clip_g", "t5xxl"),
    "CLIPTextEncodeFlux": ("clip_l", "t5xxl"),
    "CLIPTextEncodeHiDream": ("clip_l", "clip_g", "t5xxl", "llama"),
    "CLIPTextEncodeHunyuanDiT": ("bert", "mt5xl"),
    "TextEncodeQwenImageEdit": ("prompt",),
    "TextEncodeQwenImageEditPlus": ("prompt",),
    "TextEncodeJoyImageEdit": ("prompt",),
    "MiniMaxH3ImageToVideo": ("prompt",),
    "MiniMaxH3ReferenceToVideo": ("prompt",),
}
DUAL_ENCODERS = {"TextEncodeQwenImage21", "TextEncodeMageFlowEdit", "TextEncodeBooguEdit"}
ENCODER_FAMILY = {
    "CLIPTextEncodeSDXL": "SDXL", "CLIPTextEncodeSDXLRefiner": "SDXL",
    "CLIPTextEncodeSD3": "SD3 / SD3.5", "CLIPTextEncodeFlux": "FLUX.1",
    "CLIPTextEncodeHiDream": "HiDream", "CLIPTextEncodeHunyuanDiT": "HunyuanDiT",
    "TextEncodeQwenImage21": "Qwen Image 2.1", "TextEncodeQwenImageEdit": "Qwen Image Edit",
    "TextEncodeQwenImageEditPlus": "Qwen Image Edit Plus", "TextEncodeMageFlowEdit": "MageFlow",
    "TextEncodeBooguEdit": "Boogu", "TextEncodeJoyImageEdit": "JoyImage",
    "MiniMaxH3ImageToVideo": "MiniMax H3", "MiniMaxH3ReferenceToVideo": "MiniMax H3",
    "AnimaRegionalCanvas": "Anima",
}
CLIP_FAMILY = {
    "sdxl": "SDXL", "sd3": "SD3 / SD3.5", "stable_cascade": "Stable Cascade",
    "pixart": "PixArt", "flux": "FLUX.1", "flux2": "FLUX.2", "krea2": "Krea2",
    "qwen_image": "Qwen Image", "hunyuan_image": "Hunyuan Image",
    "hidream": "HiDream", "chroma": "Chroma", "longcat_image": "LongCat Image",
    "omnigen2": "OmniGen2", "boogu": "Boogu", "joyimage": "JoyImage", "mage": "MageFlow",
    "ideogram4": "Ideogram 4", "lens": "Microsoft Lens", "pixeldit": "PixelDiT",
    "kandinsky5_image": "Kandinsky 5 Image", "ovis": "Ovis Image", "newbie": "NewBie",
}
IMAGE_PATHS = {
    "ImageScale": "image", "FluxKontextImageScale": "image", "ImageScaleBy": "image", "ImageCrop": "image", "ImageInvert": "image",
    "ImagePadForOutpaint": "image", "ImageFromBatch": "image", "RepeatImageBatch": "image",
    "ImageUpscaleWithModel": "image",  # deterministic postprocess, not a diffusion producer
}
LATENT_PATHS = {
    "LatentUpscale": "samples", "LatentUpscaleBy": "samples", "LatentCrop": "samples",
    "LatentRotate": "samples", "LatentFlip": "samples", "LatentFromBatch": "samples",
    "RepeatLatentBatch": "samples", "SetLatentNoiseMask": "samples",
}
COND_PATHS = {
    "ReferenceLatent": "conditioning", "FluxGuidance": "conditioning", "FluxDisableGuidance": "conditioning",
    "MiniMaxH3AddGuide": "positive", "FluxKontextMultiReferenceLatentMethod": "conditioning", "ConditioningSetArea": "conditioning",
    "ConditioningSetAreaPercentage": "conditioning", "ConditioningSetMask": "conditioning",
    "ConditioningSetTimestepRange": "conditioning", "ConditioningSetAreaStrength": "conditioning",
    "ControlNetApply": "conditioning", "StableCascade_StageB_Conditioning": "conditioning",
    "PiDConditioning": "positive",
}
DUAL_COND = {"ControlNetApplyAdvanced", "ControlNetApplySD3", "InpaintModelConditioning", "HiDreamO1ReferenceImages"}
MODEL_PATHS = {
    "QwenImage21Cache", "MiniMaxH3SigmaShift", "MiniMaxH3FunControlNetApply",
    "ModelSamplingSD3", "ModelSamplingFlux", "ModelSamplingAuraFlow", "ModelSamplingDiscrete",
    "ModelSamplingContinuousEDM", "ModelSamplingContinuousV", "ModelSamplingStableCascade",
    "ModelSamplingDiscreteDistilled", "ModelSamplingLTXV", "FluxKVCache",
    "DifferentialDiffusion", "RescaleCFG", "ChromaRadianceOptions", "HiDreamO1PatchSeamSmoothing",
}
CLIP_LOADERS = {"CLIPLoader": ("clip_name",), "DualCLIPLoader": ("clip_name1", "clip_name2"),
                "TripleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3"),
                "QuadrupleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3", "clip_name4")}
# UI widget orders (including the frontend-only seed control). Named values win.
WIDGETS = {
    "KSampler": "seed control_after_generate steps cfg sampler_name scheduler denoise",
    "KSamplerAdvanced": "add_noise noise_seed control_after_generate steps cfg sampler_name scheduler start_at_step end_at_step return_with_leftover_noise",
    "SamplerCustom": "add_noise noise_seed control_after_generate cfg",
    "RandomNoise": "noise_seed control_after_generate", "KSamplerSelect": "sampler_name",
    "BasicScheduler": "scheduler steps denoise", "CFGGuider": "cfg",
    "DualCFGGuider": "cfg_conds cfg_cond2_negative style", "DualModelGuider": "cfg",
    "CLIPLoader": "clip_name type device", "DualCLIPLoader": "clip_name1 clip_name2 type device",
    "TripleCLIPLoader": "clip_name1 clip_name2 clip_name3", "QuadrupleCLIPLoader": "clip_name1 clip_name2 clip_name3 clip_name4",
    "UNETLoader": "unet_name weight_dtype", "CheckpointLoaderSimple": "ckpt_name",
    "CheckpointLoader": "config_name ckpt_name", "VAELoader": "vae_name",
    "LoraLoader": "lora_name strength_model strength_clip", "LoraLoaderModelOnly": "lora_name strength_model",
    "CLIPSetLastLayer": "stop_at_clip_layer", "ComfySwitchNode": "switch",
    "CLIPTextEncode": "text", "CLIPTextEncodeSDXL": "width height crop_w crop_h target_width target_height text_g text_l",
    "CLIPTextEncodeSDXLRefiner": "ascore width height text", "CLIPTextEncodeSD3": "clip_l clip_g t5xxl empty_padding",
    "CLIPTextEncodeHiDream": "clip_l clip_g t5xxl llama", "CLIPTextEncodeHunyuanDiT": "bert mt5xl",
    "CLIPTextEncodeFlux": "clip_l t5xxl guidance", "FluxGuidance": "guidance",
    "TextEncodeQwenImage21": "prompt negative_prompt resolution", "TextEncodeQwenImageEdit": "prompt",
    "TextEncodeQwenImageEditPlus": "prompt", "TextEncodeMageFlowEdit": "prompt negative_prompt width height batch_size",
    "TextEncodeBooguEdit": "prompt negative_prompt", "TextEncodeJoyImageEdit": "prompt",
    "MiniMaxH3ImageToVideo": "prompt width height length", "MiniMaxH3ReferenceToVideo": "prompt width height length ref_image_size",
    "MiniMaxH3SigmaShift": "shift_video shift_audio", "MiniMaxH3AddGuide": "frame_idx",
    "ImageFromBatch": "batch_index length", "ImageScale": "upscale_method width height crop",
    "ImageScaleBy": "upscale_method scale_by", "EmptyLatentImage": "width height batch_size",
    "EmptySD3LatentImage": "width height batch_size", "EmptyFlux2LatentImage": "width height batch_size",
    "EmptyMiniMaxH3LatentAV": "width height length", "EmptyQwenImageLayeredLatentImage": "width height layers batch_size",
    "EmptyHiDreamO1LatentImage": "width height batch_size", "EmptyChromaRadianceLatentImage": "width height batch_size",
    "ResolutionSelector": "aspect_ratio megapixels multiple", "PrimitiveString": "value",
    "PrimitiveStringMultiline": "value", "PrimitiveInt": "value control_after_generate",
    "PrimitiveFloat": "value", "PrimitiveBoolean": "value", "StringConcatenate": "string_a string_b delimiter",
    "StringSubstring": "string start end", "H3CharacterSheetDesigner": "state_json",
}
SCHEDULES = {"Flux2Scheduler": "flux2","KarrasScheduler": "karras", "ExponentialScheduler": "exponential",
             "PolyexponentialScheduler": "polyexponential", "LaplaceScheduler": "laplace",
             "VPScheduler": "vp", "SDTurboScheduler": "sd_turbo", "BetaSamplingScheduler": "beta"}
FIXED_SAMPLERS = {"SamplerEulerAncestral": "euler_ancestral", "SamplerEulerAncestralCFGPP": "euler_ancestral_cfg_pp",
                  "SamplerLMS": "lms", "SamplerDPMAdaptative": "dpm_adaptive", "SamplerER_SDE": "er_sde",
                  "SamplerSASolver": "sa_solver", "SamplerSEEDS2": "seeds_2", "SamplerDPMPP_2S_Ancestral": "dpmpp_2s_ancestral"}
# Paired output is not inferred from the caller's 'negative' flag.
PROVIDER_ROLES = {
    "H3CharacterSheetDesigner": ("string", "scalar", "scalar"),
    "Krea2ElementFramingV1Canvas": ("framing",),
    "Krea2ElementFramingV1Prompt": ("prompt_data",),
    "Krea2ElementJSONExportV1": ("string", "scalar", "scalar"),
    "Krea2BBOXPromptEffect": ("string", "string"), "Krea2PromptEffect": ("string", "string"),
    "Krea2BackgroundEffect": ("string", "string"),
    "Krea2BBOXPoseGestureInjectorV1": ("prompt_data", "string", "string"),
}

WIDGETS.update({"Flux2Scheduler": "steps width height", "KarrasScheduler": "steps sigma_max sigma_min rho",
    "ExponentialScheduler": "steps sigma_max sigma_min", "PolyexponentialScheduler": "steps sigma_max sigma_min rho",
    "LaplaceScheduler": "steps sigma_max sigma_min mu beta", "VPScheduler": "steps beta_d beta_min eps_s",
    "SDTurboScheduler": "steps denoise", "BetaSamplingScheduler": "steps alpha beta"})
