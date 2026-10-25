"""One training/evaluation lifecycle for every preset and custom composition."""

import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import checkpoints
from .artifacts import environment, seal, sha256, write_json
from .data.dataset import CQTTracks
from .data.manifests import load_partition, validate_partitions
from .evaluation import embed, retrieval_metrics
from .losses.compose import Objectives
from .mining.dynamic_nsim import NegativeCache, due
from .models.adapters import ExperimentModel
from .sampling.balanced import BalancedSampler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def labels_for(rows, config):
    labels = sorted({r["clique_id"] for r in rows})
    if config["profile"] == "upstream":
        if any(not v.isdecimal() or not 0 <= int(v) < 10000 for v in labels):
            raise ValueError("Upstream CE requires original numeric clique IDs in [0, 9999]")
        return {v: int(v) for v in labels}, 10000
    return {value: index for index, value in enumerate(labels)}, len(labels)


def preflight(resolved, init=None, resume=None, training=True):
    config = resolved["config"]
    if init and resume:
        raise ValueError("Warm start and resume are mutually exclusive")
    if resolved["dependency"] and not (init or resume):
        raise ValueError(f"{resolved['preset']} requires an explicit --init checkpoint from {resolved['dependency']} (or --resume)")
    if config["train"]["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    report = validate_partitions(config)
    train_rows = load_partition(config, "train")
    mapping, classes = labels_for(train_rows, config)
    # Materialize validation, cache and sampler contracts before allocating a run.
    dataset = CQTTracks(train_rows, config, labels=mapping)
    CQTTracks(load_partition(config, "val"), config)
    for name in config["evaluation"]["sets"]:
        data, modality = name.split(".")
        CQTTracks(load_partition(config, "test", data, modality), config)
    if config["objectives"]["modified_triplet"]["weight"] and len(set(dataset.labels)) < 2:
        raise ValueError("Modified triplet requires at least two training cliques")
    if config["sampler"]["kind"] == "balanced":
        sampler = BalancedSampler(dataset.labels, dataset.originals, config["sampler"], config["train"]["seed"])
        if not len(sampler):
            raise ValueError("Balanced sampler produces no batches")
    if training and config["train"]["batch_size"] > len(dataset) and config["sampler"]["kind"] == "random":
        raise ValueError("Training partition is smaller than one batch")
    if init or resume:
        state = checkpoints.load(init or resume)
        model = ExperimentModel(classes, config["model"])
        if resume:
            checkpoints.require_resume_state(state)
            validate_resume_config(config, state["config"])
            model.load_state_dict(state["model"], strict=True)
            objectives = Objectives(config, classes)
            objectives.load_state_dict(state["objectives"], strict=True)
        else:
            if "label_map" in state and state["label_map"] != mapping and not config["model"]["replace_classifier"]:
                raise ValueError("Initializer label map differs; explicitly enable model.replace_classifier")
            report["checkpoint_events"] = checkpoints.warm_start(model, state, config["model"]["replace_classifier"])
        report["initializer"] = {"path": str(Path(init or resume).resolve()), "sha256": sha256(init or resume), "mode": "resume" if resume else "warm_start"}
    report["classes"] = classes
    return report


def validate_resume_config(config, previous):
    if config != previous:
        raise ValueError("Resume requires the identical resolved configuration, including scheduler horizon; use --init for a new experiment")


def training_loaders(rows, config, mapping):
    frames = (200, 300, 400) if config["profile"] == "upstream" else (None,)
    loaders, sampler = [], None
    for length in frames:
        dataset = CQTTracks(rows, config, training=True, frames=length, labels=mapping)
        options = {"num_workers": config["train"]["workers"], "worker_init_fn": worker_seed}
        if config["sampler"]["kind"] == "balanced":
            sampler = BalancedSampler(dataset.labels, dataset.originals, config["sampler"], config["train"]["seed"])
            loader = DataLoader(dataset, batch_sampler=sampler, **options)
        else:
            loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, drop_last=False, **options)
        loaders.append(loader)
    return loaders, sampler


def evaluate_sets(model, config, device, run):
    results = {}
    for name in config["evaluation"]["sets"]:
        dataset_name, modality = name.split(".")
        rows = load_partition(config, "test", dataset_name, modality)
        dataset = CQTTracks(rows, config)
        features = embed(model, dataset, device, workers=config["train"]["workers"])
        result = retrieval_metrics(features.numpy(), dataset.labels, config["evaluation"]["no_positive"], [r["track_id"] for r in rows])
        result["track_order"] = [r["track_id"] for r in rows]
        write_json(Path(run) / "evaluation" / f"{name}.json", result)
        results[name] = {k: v for k, v in result.items() if k not in ("rankings", "per_query", "track_order")}
    write_json(Path(run) / "metrics.json", results)
    return results


def train(resolved, run_id, init=None, resume=None, stop_after=None):
    config = resolved["config"]
    settings = config["train"]
    seed_everything(settings["seed"])
    report = preflight(resolved, init, resume)
    # Preflight may initialize a model to validate checkpoints; keep that RNG use
    # out of the actual experiment's initialization.
    seed_everything(settings["seed"])
    if Path(run_id).name != run_id or run_id in ("", ".", ".."):
        raise ValueError("run-id must be a single directory name")
    run = Path(config["paths"]["runs"]) / run_id
    if run.exists():
        raise ValueError(f"Run already exists; choose a new run-id: {run}")
    rows = load_partition(config, "train")
    mapping, classes = labels_for(rows, config)
    model = ExperimentModel(classes, config["model"]).to(settings["device"])
    objectives = Objectives(config, classes).to(settings["device"])
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), objectives.parameters()), lr=settings["lr"], weight_decay=settings["weight_decay"])
    if settings["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, settings["epochs"]), eta_min=settings["min_lr"])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=settings["plateau_factor"],
            patience=settings["plateau_patience"], min_lr=settings["min_lr"],
        )
    loaders, sampler = training_loaders(rows, config, mapping)
    bank_dataset = CQTTracks(rows, config, labels=mapping)
    validation = CQTTracks(load_partition(config, "val"), config)
    cache = NegativeCache()
    epoch_start, best, threshold_best, stale, best_epoch = 0, -1.0, -1.0, 0, -1
    previous_best, best_objectives = None, None
    lineage = report.get("initializer")
    if init:
        state = checkpoints.load(init)
        events = checkpoints.warm_start(model, state, config["model"]["replace_classifier"])
        if "objectives" in state and objectives.arcface is not None:
            # Warm start transfers matching auxiliary heads explicitly; the
            # optimizer, schedule, mining and RNG always start fresh.
            source_head = {k: v for k, v in state["objectives"].items() if k.startswith("arcface.")}
            if source_head and state.get("label_map") == mapping:
                objectives.load_state_dict(source_head, strict=True)
                events.append("Transferred ArcFace head")
            elif source_head:
                events.append("ArcFace head reinitialized: class identity differs or is unverified")
        report["checkpoint_events"] = events
    if resume:
        state = checkpoints.load(resume)
        validate_resume_config(config, state["config"])
        model.load_state_dict(state["model"], strict=True)
        objectives.load_state_dict(state["objectives"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cache.load_state_dict(state["negative_cache"])
        if sampler:
            sampler.load_state_dict(state["sampler"])
        epoch_start, best, threshold_best = state["next_epoch"], state["best"], state["threshold_best"]
        stale, best_epoch = state["stale"], state["best_epoch"]
        previous_best = state["best_model"]
        best_objectives = state["best_objectives"]
        if state["input_hashes"] != report["manifest_hashes"]:
            raise ValueError("Resume manifests changed")
        if state["source_hashes"] != environment()["source_hashes"]:
            raise ValueError("Resume implementation changed; use --init for a derived experiment")
        restore = state["rng"]
        checkpoints.restore_rng(restore)
    if settings["epochs"] == 0 and not init:
        raise ValueError("Zero-epoch evaluation requires --init")
    run.mkdir(parents=True)
    write_json(run / "config.json", resolved)
    provenance = {"environment": environment(), "inputs": report, "initializer": lineage, "historical_reproduction": False}
    write_json(run / "provenance.json", provenance)
    write_json(run / "label_map.json", mapping)
    source_hashes = provenance["environment"]["source_hashes"]
    history = run / "history.jsonl"
    if resume:
        history.write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in state["history"]))
    else:
        history.touch()
    history_values = list(state["history"]) if resume else []

    def snapshot(next_epoch):
        return {
            "format": 1, "config": config, "model": model.state_dict(),
            "kind": "training",
            "objectives": objectives.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": checkpoints.rng_state(),
            "sampler": sampler.state_dict() if sampler else None,
            "negative_cache": cache.state_dict(), "next_epoch": next_epoch,
            "best": best, "threshold_best": threshold_best, "stale": stale,
            "best_epoch": best_epoch, "best_model": previous_best,
            "best_objectives": best_objectives,
            "history": history_values, "input_hashes": report["manifest_hashes"],
            "source_hashes": source_hashes, "initializer": lineage,
            "label_map": mapping,
        }

    completed = epoch_start
    for epoch in range(epoch_start, settings["epochs"]):
        if settings["patience"] and stale >= settings["patience"]:
            break
        model.train()
        objectives.train()
        if sampler:
            sampler.epoch = epoch
        refresh_events = []
        if config["objectives"]["modified_triplet"]["weight"] and (
            due(epoch, cache.selection_epoch, config["negative"]["selection_refresh_epochs"])
            or due(epoch, cache.embedding_epoch, config["negative"]["embedding_refresh_epochs"])
        ):
            bank = embed(model, bank_dataset, settings["device"], workers=settings["workers"])
            refresh_events = cache.refresh(epoch, bank, torch.tensor(bank_dataset.labels), config["negative"])
        batch_logs, loss_sum, examples, updates = [], 0.0, 0, 0
        for group in zip(*loaders):
            for batch in group:
                labels = batch["label"].to(settings["device"])
                originals = batch["original"].to(settings["device"])
                identities = batch["index"].to(settings["device"])
                logits, embeddings = model(batch["cqt"].to(settings["device"]))
                negatives = None
                if config["objectives"]["modified_triplet"]["weight"]:
                    indices = batch["index"]
                    similar_indices = cache.identities[indices]
                    all_labels = torch.tensor(bank_dataset.labels)
                    negative_mask = all_labels[indices, None] != all_labels[None, :]
                    random_indices = torch.multinomial(negative_mask.float(), 1).squeeze(1)
                    if config["negative"]["embedding_policy"] == "cached":
                        similar = cache.embeddings[similar_indices].to(settings["device"])
                        random_negative = cache.embeddings[random_indices].to(settings["device"])
                    else:
                        data = torch.stack([bank_dataset[int(i)]["cqt"] for i in torch.cat((similar_indices, random_indices))])
                        _, negative_features = model(data.to(settings["device"]))
                        similar, random_negative = negative_features.chunk(2)
                    negatives = (similar, random_negative)
                optimizer.zero_grad(set_to_none=True)
                loss, logs = objectives(logits, embeddings, labels, originals, identities, negatives)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at epoch {epoch}")
                if not logs["skipped"]:
                    loss.backward()
                    optimizer.step()
                    updates += 1
                batch_logs.append(logs)
                loss_sum += float(loss.detach())
                examples += len(labels)
        if not updates:
            raise ValueError(f"Epoch {epoch} produced no valid optimizer updates; check anchors, positives and sampling")
        features = embed(model, validation, settings["device"], workers=settings["workers"])
        metrics = retrieval_metrics(features.numpy(), validation.labels, config["evaluation"]["no_positive"])
        score = metrics["MAP"]
        if score > best:
            best, best_epoch = score, epoch
            previous_best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_objectives = {key: value.detach().cpu().clone() for key, value in objectives.state_dict().items()}
        if score > threshold_best + settings["min_improvement"]:
            threshold_best, stale = score, 0
        else:
            stale += 1
        # Preserve upstream's sum(batch mean CE)/number of examples statistic.
        schedule_loss = loss_sum / examples
        if settings["scheduler"] == "plateau":
            scheduler.step(schedule_loss)
        else:
            scheduler.step()
        record = {
            "epoch": epoch, "loss": loss_sum / len(batch_logs), "upstream_schedule_loss": schedule_loss,
            "validation_MAP": score, "best_epoch": best_epoch, "stale": stale,
            "lr": optimizer.param_groups[0]["lr"], "refresh": refresh_events,
            "batches": batch_logs, "updates": updates,
        }
        history_values.append(record)
        with history.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        completed = epoch + 1
        saved = snapshot(completed)
        checkpoints.save(run / "checkpoints" / "last.pt", saved)
        if stop_after is not None and completed >= stop_after:
            break
    if completed == 0:
        previous_best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_objectives = {key: value.detach().cpu().clone() for key, value in objectives.state_dict().items()}
        checkpoints.save(run / "checkpoints" / "last.pt", snapshot(0))
    elif not (run / "checkpoints" / "last.pt").exists():
        checkpoints.save(run / "checkpoints" / "last.pt", snapshot(completed))
    checkpoints.save(run / "checkpoints" / "best.pt", {
        "format": 1, "kind": "evaluation", "model": previous_best, "objectives": best_objectives,
        "config": config, "best_epoch": best_epoch, "label_map": mapping, "source_hashes": source_hashes,
    })
    model.load_state_dict(previous_best, strict=True)
    results = evaluate_sets(model, config, settings["device"], run)
    write_json(run / "status.json", {"completed_epochs": completed, "budget": settings["epochs"], "best_epoch": best_epoch, "best_MAP": best, "interrupted": stop_after is not None and completed < settings["epochs"]})
    seal(run)
    return {"run": str(run), "completed_epochs": completed, "best_epoch": best_epoch, "metrics": results}


def evaluate(resolved, checkpoint, run_id):
    if not checkpoint:
        raise ValueError("Evaluation requires --init CHECKPOINT")
    config = resolved["config"]
    state = checkpoints.load(checkpoint)
    weights = state.get("model", state.get("state_dict", state))
    classes = next((value for key, value in weights.items() if key.removeprefix("module.").removeprefix("backbone.") == "fc1.bias"), None)
    if classes is None:
        raise ValueError("Checkpoint has no classifier")
    model = ExperimentModel(len(classes), config["model"]).to(config["train"]["device"])
    checkpoints.warm_start(model, state, replace_classifier=False)
    run = Path(config["paths"]["runs"]) / run_id
    if Path(run_id).name != run_id or run.exists():
        raise ValueError("Evaluation requires a fresh single-component run-id")
    # Validate every requested set before creating output.
    input_hashes = {}
    for name in config["evaluation"]["sets"]:
        dataset, modality = name.split(".")
        CQTTracks(load_partition(config, "test", dataset, modality), config)
        from .data.manifests import partition_path
        path = partition_path(config, "test", dataset)
        input_hashes[str(path)] = sha256(path)
    run.mkdir(parents=True)
    write_json(run / "config.json", resolved)
    write_json(run / "provenance.json", {
        "environment": environment(), "checkpoint": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)},
        "inputs": {"manifest_hashes": input_hashes},
        "preprocessing_matches_checkpoint": state.get("config", {}).get("preprocess") == config["preprocess"],
        "checkpoint_events": "Strict backbone/classifier load; requested preprocessing recorded independently.",
    })
    result = evaluate_sets(model, config, config["train"]["device"], run)
    seal(run)
    return {"run": str(run), "metrics": result}


def dispatch(args, resolved):
    if args.command == "doctor":
        return preflight(resolved, args.init, args.resume)
    if args.command == "train":
        return train(resolved, args.run_id, args.init, args.resume)
    if args.command == "evaluate":
        if args.resume:
            raise ValueError("evaluate accepts --init, not --resume")
        return evaluate(resolved, args.init, args.run_id)
    raise ValueError(f"Unsupported lifecycle command: {args.command}")
