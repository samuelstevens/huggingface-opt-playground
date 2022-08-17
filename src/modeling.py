import os

import accelerate
import huggingface_hub
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import helpers

CACHE_DIR = "/research/nfs_su_809/huggingface_cache"


def load_tokenizer(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)


def _load_opt_with_decoder(weights_path, model):
    device_map = accelerate.infer_auto_device_map(
        model.model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16"
    )

    with helpers.logged("Loading checkpoint"):
        accelerate.load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            dtype="float16",
            offload_state_dict=True,
        )
        model.tie_weights()

        # https://github.com/huggingface/accelerate/issues/362
        full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
        full_model_device_map["lm_head"] = 0
        accelerate.dispatch_model(model, device_map=full_model_device_map)

    return model


def _load_opt_without_decoder(weights_path, model):
    device_map = accelerate.infer_auto_device_map(
        model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16"
    )

    with helpers.logged("Loading checkpoint"):
        accelerate.load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map=device_map,
            dtype="float16",
            offload_state_dict=True,
        )
        model.tie_weights()
    return model


def _load_opt(weights_path, checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)

    # Initializes an empty shell with the model.
    # This is instant and does not take any RAM.
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    # Initializing the model under the previous context
    # manager breaks the tied weights.
    model.tie_weights()

    if "13b" in checkpoint:
        return _load_opt_with_decoder(weights_path, model)
    elif "6.7b" in checkpoint:
        return _load_opt_with_decoder(weights_path, model)
    elif "350m" in checkpoint:
        return _load_opt_with_decoder(weights_path, model)
    
    # Some of the checkpoitns have a decoder. Some of them don't. (:
    if "2.7b" in checkpoint:
        return _load_opt_without_decoder(weights_path, model)
    elif "1.3b" in checkpoint:
        return _load_opt_without_decoder(weights_path, model)
    elif "125m" in checkpoint:
        return _load_opt_without_decoder(weights_path, model)

    raise ValueError(checkpoint)


def _load_opt_30b(weights_path):
    config = AutoConfig.from_pretrained("facebook/opt-30b")

    # Initializes an empty shell with the model.
    # This is instant and does not take any RAM.
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    # Initializing the model under the previous context
    # manager breaks the tied weights.
    model.tie_weights()

    with helpers.logged("Inferring device map"):
        device_map = {
            "decoder.embed_tokens": 0,
            "decoder.embed_positions": 0,
            "decoder.final_layer_norm": 0,
            "decoder.layers.0": 0,
            "decoder.layers.1": 0,
            "decoder.layers.2": 0,
            "decoder.layers.3": 0,
            "decoder.layers.4": 0,
            "decoder.layers.5": 0,
            "decoder.layers.6": 0,
            "decoder.layers.7": 0,
            "decoder.layers.8": 0,
            "decoder.layers.9": 0,
            "decoder.layers.10": 0,
            "decoder.layers.11": 0,
            "decoder.layers.12": 0,
            "decoder.layers.13": 0,
            "decoder.layers.14": 0,
            "decoder.layers.15": 0,
            "decoder.layers.16": 0,
            "decoder.layers.17": 0,
            "decoder.layers.18": 0,
            "decoder.layers.19": 0,
            "decoder.layers.20": 1,
            "decoder.layers.21": 1,
            "decoder.layers.22": 1,
            "decoder.layers.23": 1,
            "decoder.layers.24": 1,
            "decoder.layers.25": 1,
            "decoder.layers.26": 1,
            "decoder.layers.27": 1,
            "decoder.layers.28": 1,
            "decoder.layers.29": 1,
            "decoder.layers.30": 1,
            "decoder.layers.31": 1,
            "decoder.layers.32": 1,
            "decoder.layers.33": 1,
            "decoder.layers.34": 1,
            "decoder.layers.35": 1,
            "decoder.layers.36": 1,
            "decoder.layers.37": 1,
            "decoder.layers.38": 1,
            "decoder.layers.39": 1,
            "decoder.layers.40": 1,
            "decoder.layers.41": 1,
            "decoder.layers.42": 1,
            "decoder.layers.43": 1,
            "decoder.layers.44": 1,
            "decoder.layers.45": 1,
            "decoder.layers.46": 1,
            "decoder.layers.47": 1,
        }

        helpers.log("device map:", device_map)

    with helpers.logged("Loading checkpoint"):
        accelerate.load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            dtype="float16",
            offload_state_dict=True,
        )
        model.tie_weights()

        # https://github.com/huggingface/accelerate/issues/362
        full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
        full_model_device_map["lm_head"] = 0
        accelerate.dispatch_model(model, device_map=full_model_device_map)

    return model


def _load_gpt2(checkpoint):
    lm_config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, config=lm_config)

    return model.to(torch.device("cuda:0"))


def load_model(checkpoint):
    with helpers.logged(f"Downloading {checkpoint}"):
        weights_path = huggingface_hub.snapshot_download(
            checkpoint,
            cache_dir=CACHE_DIR,
            ignore_regex=[r"tf_model.*", r"flax_model.*"],
        )

    # If the folder contains a checkpoint that isn't sharded,
    # it needs to point to the state dict directly otherwise
    # point to the directory containing the shard
    files = os.listdir(weights_path)
    weights_path = (
        os.path.join(weights_path, "pytorch_model.bin")
        if "pytorch_model.bin" in files
        else weights_path
    )

    if "gpt2" in checkpoint:
        return _load_gpt2(checkpoint)
    elif checkpoint == "facebook/opt-30b":
        return _load_opt_30b(weights_path)
    elif "facebook/opt" in checkpoint:
        return _load_opt(weights_path, checkpoint)

    raise ValueError(checkpoint)
