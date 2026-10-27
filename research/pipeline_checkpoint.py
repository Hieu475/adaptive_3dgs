"""Portable, verified checkpoints for replaying a mapping pipeline snapshot.

These checkpoints are for controlled experiment replay. They include optimizer,
tracker, scheduler, importance-estimator, Gaussian identity, cache, and RNG state;
they are not a general training-resume format.
"""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCHEMA = "online-mapping-replay-checkpoint-v1"
PARAMETER_NAMES = (
    "_xyz", "_scaling", "_rotation", "_opacity", "_features_dc",
    "_features_rest", "_normals",
)


def config_digest(config) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def gaussian_map_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    for name, value in sorted(vars(model.state_store).items()):
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _encode(value):
    if torch.is_tensor(value):
        return {"kind": "tensor", "value": value.detach().contiguous().cpu().clone()}
    if isinstance(value, np.ndarray):
        return {"kind": "ndarray", "value": torch.from_numpy(value.copy())}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "primitive", "value": value}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [(_encode(k), _encode(v)) for k, v in value.items()]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(v) for v in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(v) for v in value]}
    if isinstance(value, set):
        return {"kind": "set", "items": [_encode(v) for v in sorted(value, key=repr)]}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            "kind": "object",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "attrs": _encode(vars(value)),
        }
    raise TypeError(f"Unsupported checkpoint value: {type(value).__module__}.{type(value).__qualname__}")


def _decode(value, device):
    kind = value["kind"]
    if kind == "tensor":
        return value["value"].to(device)
    if kind == "ndarray":
        return value["value"].cpu().numpy().copy()
    if kind == "primitive":
        return value["value"]
    if kind == "dict":
        return {_decode(k, device): _decode(v, device) for k, v in value["items"]}
    if kind == "list":
        return [_decode(v, device) for v in value["items"]]
    if kind == "tuple":
        return tuple(_decode(v, device) for v in value["items"])
    if kind == "set":
        return {_decode(v, device) for v in value["items"]}
    raise TypeError(f"Cannot decode {kind!r} as a detached value")


def _restore_object(target, value, device):
    if value["kind"] != "object":
        raise TypeError("Expected an encoded object")
    target_class = f"{type(target).__module__}.{type(target).__qualname__}"
    if target_class != value["class"]:
        raise TypeError(f"Checkpoint component type mismatch: {target_class} != {value['class']}")
    attrs = value["attrs"]
    if attrs["kind"] != "dict":
        raise TypeError("Malformed object attributes")
    for encoded_key, encoded_value in attrs["items"]:
        key = _decode(encoded_key, device)
        current = getattr(target, key, None)
        if encoded_value["kind"] == "object":
            if current is None:
                raise TypeError(f"No existing compatible object for {target_class}.{key}")
            _restore_object(current, encoded_value, device)
        else:
            setattr(target, key, _decode(encoded_value, device))


def _rng_state(device):
    return {
        "python": _encode(random.getstate()),
        "numpy": _encode(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()]
        if str(device).startswith("cuda") and torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(_decode(state["python"], "cpu"))
    np.random.set_state(_decode(state["numpy"], "cpu"))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from replay checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture_pipeline_checkpoint(pipeline, *, scene, seed, snapshot_frame):
    """Capture a complete frame-boundary replay state on CPU."""
    if not pipeline.initialized or pipeline.optimizer is None:
        raise RuntimeError("A replay checkpoint requires an initialized pipeline and optimizer")
    if pipeline._custom_selector_fn is not None or pipeline._pre_scheduling_hook is not None:
        raise RuntimeError("Cannot checkpoint process-local selector or scheduling hooks")

    model = pipeline.gaussian_model
    model_state = {name: tensor.detach().contiguous().cpu().clone()
                   for name, tensor in model.state_dict().items()}
    optimizer_state = {}
    for name in PARAMETER_NAMES:
        parameter = getattr(model, name)
        state = pipeline.optimizer.state.get(parameter)
        optimizer_state[name] = None if state is None else {
            key: tensor.detach().contiguous().cpu().clone() for key, tensor in state.items()
        }
    optimizer_groups = [{key: copy.deepcopy(value) for key, value in group.items()
                         if key != "params"}
                        for group in pipeline.optimizer.param_groups]

    components = {
        "tracker": _encode(pipeline.tracker),
        "importance_estimator": _encode(pipeline.importance_estimator),
        "scheduler": _encode(pipeline.scheduler),
        "background_cache": _encode(pipeline.bg_cache),
    }
    utility_state = (None if pipeline.utility_predictor is None else {
        key: tensor.detach().contiguous().cpu().clone()
        for key, tensor in pipeline.utility_predictor.state_dict().items()
    })
    return {
        "schema": SCHEMA,
        "metadata": {
            "scene": str(scene), "seed": int(seed),
            "snapshot_frame": int(snapshot_frame),
            "config_sha256": config_digest(pipeline.config),
            "map_digest": gaussian_map_digest(pipeline.gaussian_model),
            "n_gaussians": int(model.num_gaussians),
            "frame_count": int(pipeline.frame_count),
        },
        "gaussian_model_state_dict": model_state,
        "gaussian_state_store": _encode(model.state_store.state_dict()),
        "optimizer": {
            "betas": tuple(pipeline.optimizer.betas), "eps": float(pipeline.optimizer.eps),
            "groups": optimizer_groups, "state_by_parameter": optimizer_state,
        },
        "components": components,
        "pipeline_state": _encode({
            "frame_count": pipeline.frame_count,
            "spawn_control_count": pipeline._spawn_control_count,
            "current_pose": pipeline.current_pose,
            "intrinsics": pipeline.intrinsics,
            "initialized": pipeline.initialized,
            "metrics_history": pipeline.metrics_history,
            "repro_trace": pipeline._repro_trace,
            "init_repro_trace": pipeline._init_repro_trace,
            "last_optimize_mask": getattr(pipeline, "_last_optimize_mask", None),
            "learned_utility_scores": pipeline._learned_utility_scores,
            "predicted_delta_q": pipeline._predicted_delta_q,
            "predicted_delta_t": pipeline._predicted_delta_t,
        }),
        "utility_predictor_state_dict": utility_state,
        "rng_state": _rng_state(pipeline.device),
    }


def save_pipeline_checkpoint(path, pipeline, *, scene, seed, snapshot_frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = capture_pipeline_checkpoint(
        pipeline, scene=scene, seed=seed, snapshot_frame=snapshot_frame)
    torch.save(payload, path)
    return payload["metadata"]


def restore_pipeline_checkpoint(pipeline, payload, *, scene=None, seed=None,
                                snapshot_frame=None, restore_rng=True):
    """Restore a replay checkpoint into a freshly constructed pipeline."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported replay checkpoint schema: {payload.get('schema')}")
    metadata = payload["metadata"]
    expected = {"scene": scene, "seed": seed, "snapshot_frame": snapshot_frame}
    mismatches = {key: (value, metadata.get(key)) for key, value in expected.items()
                  if value is not None and metadata.get(key) != value}
    if config_digest(pipeline.config) != metadata.get("config_sha256"):
        mismatches["config_sha256"] = (config_digest(pipeline.config), metadata.get("config_sha256"))
    if mismatches:
        raise ValueError(f"Replay checkpoint metadata/config mismatch: {mismatches}")

    model = pipeline.gaussian_model
    device = pipeline.device
    for name in PARAMETER_NAMES:
        setattr(model, name, nn.Parameter(
            payload["gaussian_model_state_dict"][name].to(device).detach().clone()))
    for name in ("_confidence", "_state"):
        setattr(model, name, payload["gaussian_model_state_dict"][name].to(device).clone())
    model.state_store.load_state_dict(_decode(payload["gaussian_state_store"], device))

    pipeline._setup_optimizer()
    optimizer = payload["optimizer"]
    pipeline.optimizer.betas = tuple(optimizer["betas"])
    pipeline.optimizer.eps = float(optimizer["eps"])
    if len(pipeline.optimizer.param_groups) != len(optimizer["groups"]):
        raise ValueError("Replay checkpoint optimizer group count mismatch")
    for live_group, saved_group in zip(pipeline.optimizer.param_groups, optimizer["groups"]):
        for key, value in saved_group.items():
            live_group[key] = copy.deepcopy(value)
    for name in PARAMETER_NAMES:
        saved = optimizer["state_by_parameter"].get(name)
        if saved is not None:
            parameter = getattr(model, name)
            pipeline.optimizer.state[parameter] = {
                key: tensor.to(device).clone() for key, tensor in saved.items()
            }

    for name, encoded in payload["components"].items():
        target = {"tracker": pipeline.tracker,
                  "importance_estimator": pipeline.importance_estimator,
                  "scheduler": pipeline.scheduler,
                  "background_cache": pipeline.bg_cache}[name]
        _restore_object(target, encoded, device)
    state = _decode(payload["pipeline_state"], device)
    pipeline.frame_count = int(state["frame_count"])
    pipeline._spawn_control_count = int(state["spawn_control_count"])
    pipeline.current_pose = state["current_pose"]
    pipeline.intrinsics = state["intrinsics"]
    pipeline.initialized = bool(state["initialized"])
    pipeline.metrics_history = state["metrics_history"]
    pipeline._repro_trace = state["repro_trace"]
    pipeline._init_repro_trace = state["init_repro_trace"]
    pipeline._last_optimize_mask = state["last_optimize_mask"]
    pipeline._learned_utility_scores = state["learned_utility_scores"]
    pipeline._predicted_delta_q = state["predicted_delta_q"]
    pipeline._predicted_delta_t = state["predicted_delta_t"]
    if payload["utility_predictor_state_dict"] is not None:
        if pipeline.utility_predictor is None:
            raise ValueError("Checkpoint has utility predictor state but pipeline config did not create one")
        pipeline.utility_predictor.load_state_dict(payload["utility_predictor_state_dict"])
    if int(model.num_gaussians) != int(metadata["n_gaussians"]):
        raise RuntimeError("Restored Gaussian count does not match checkpoint metadata")
    if gaussian_map_digest(model) != metadata.get("map_digest"):
        raise RuntimeError("Restored map digest does not match checkpoint metadata")
    if restore_rng:
        _restore_rng(payload["rng_state"])
    return metadata


def load_pipeline_checkpoint(path, pipeline, *, scene=None, seed=None,
                             snapshot_frame=None, restore_rng=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return restore_pipeline_checkpoint(
        pipeline, payload, scene=scene, seed=seed,
        snapshot_frame=snapshot_frame, restore_rng=restore_rng)
