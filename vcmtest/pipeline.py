"""Work planning: which (sequence, QP, task, model) combinations to run, and which are skipped."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    IMPLEMENTED_TASKS,
    TASK_ANNOTATION_TYPE,
    model_id,
    resolve_path,
    resolve_sequence_models,
    sequence_class_ids,
    sequence_depth_min_mm,
    sequence_lpr_iou,
    sequence_obb_class_ids,
    sequence_obb_score,
    sequence_qps,
    sequence_remove_depth_maps,
    sequence_tasks,
    task_class_ids,
)
from .paths import (
    AnnotationInventory,
    DiscoveryError,
    QPItem,
    bitstream_name_warnings,
    discover_qps,
    discover_sequences,
    find_annotations,
    resolve_framerate,
    sequence_name,
)
from .score import class_names

log = logging.getLogger("vcmtest")


@dataclass
class TaskPlan:
    """One task of a sequence plan: its models, and whether it runs or is skipped (with the reason)."""
    task: str
    annotation_type: str
    models: list[dict]
    status: str              # "run" | "skipped"
    reason: str = ""
    skipped_models: dict = field(default_factory=dict)   # {configured model id: reason} (not run on this sequence)


@dataclass
class SequencePlan:
    """Work plan of one sequence: QPs, annotations, frame rate, tasks, class selections, warnings, error."""
    name: str
    dir: Path
    sequence_name: str
    qps: list[QPItem] = field(default_factory=list)
    annotations: AnnotationInventory | None = None
    fps: float | None = None
    fps_source: str | None = None
    tasks: list[TaskPlan] = field(default_factory=list)
    qp_source: str = "qps"                  # which setting selected the QPs (--qp / override / qps)
    classes: list[int] | dict | None = None # sequence_overrides.<seq>.classes ids (list, or {task: ids}); None = all
                                            # (task_classes(task) gives one task's ids)
    obb_classes: list[int] | None = None    # oriented_detection: DOTA ids (sequence_overrides.<seq>.obb_classes)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def task_classes(self, task: str) -> list[int] | None:
        """Class ids selected for one task (None = all classes)."""
        return task_class_ids(self.classes, task)

    @property
    def results_dir(self) -> Path:
        """The sequence's results folder."""
        return self.dir / "results"

    @property
    def needs_decode(self) -> bool:
        """True if any QP has no decoded PNGs yet."""
        return any(not q.has_pngs for q in self.qps)


def _plan_tasks(cfg: dict, inv: AnnotationInventory, seq: str) -> tuple[list[TaskPlan], list[str]]:
    out, warnings = [], []
    only = sequence_tasks(cfg, seq)
    for t in cfg["tasks"]:
        name = t["task"]
        ann = TASK_ANNOTATION_TYPE[name]
        models, skipped = resolve_sequence_models(cfg, seq, t)
        tp = TaskPlan(task=name, annotation_type=ann, models=models, status="run", skipped_models=skipped)
        warnings += [f"{name}/{mid} not run: {why}" for mid, why in skipped.items()]
        if only is not None and name not in only:
            tp.status, tp.reason = "skipped", f"not in sequence_overrides.{seq}.tasks {only}"
        elif not inv.has(ann):
            tp.status, tp.reason = "skipped", f"no '{ann}' annotations in {inv.folder}"
        if tp.status == "run" and name not in IMPLEMENTED_TASKS:
            tp.status, tp.reason = "skipped", f"task '{name}' not implemented yet"
        out.append(tp)
    return out, warnings


def build_plan(cfg: dict, seq_selection=None, qp_selection=None) -> list[SequencePlan]:
    """Discover the selected sequences, their QPs, annotations, frame rate and tasks.

    Errors of one sequence are stored in its plan (error) instead of being raised.
    """
    data_root = resolve_path(cfg["data_root"])
    ann_root = resolve_path(cfg["annotations_root"])
    seqs = discover_sequences(data_root, seq_selection or cfg["sequences"])
    plans = []
    for seq_dir in seqs:
        sp = SequencePlan(name=seq_dir.name, dir=seq_dir, sequence_name=sequence_name(seq_dir.name))
        plans.append(sp)
        try:
            sel, sp.qp_source = sequence_qps(cfg, sp.name, qp_selection)
            sp.qps = discover_qps(seq_dir, sel)
            sp.warnings += bitstream_name_warnings(sp.name, sp.qps)
            override = (cfg["sequence_overrides"] or {}).get(sp.name, {}).get("annotations")
            sp.annotations = find_annotations(ann_root, sp.name, override)
            for u in sp.annotations.unknown:
                hint = (f" (mask folder must be named {sp.annotations.name}_masks/)"
                        if u.is_dir() and u.name == "masks" else "")
                sp.warnings.append(f"Unrecognised annotation entry ignored: {u}{hint}")
            sp.fps, sp.fps_source, w = resolve_framerate(seq_dir, cfg, sp.qps)
            sp.warnings += w
            sp.tasks, w = _plan_tasks(cfg, sp.annotations, sp.name)
            sp.warnings += w
            sp.classes = sequence_class_ids(cfg, sp.name)
            sp.obb_classes = sequence_obb_class_ids(cfg, sp.name)
        except DiscoveryError as e:
            sp.error = str(e)
    return plans


def describe(plans: list[SequencePlan]) -> str:
    """Human-readable work plan of the sequences (as logged and printed with --plan)."""
    lines = []
    for sp in plans:
        lines.append(f"Sequence folder {sp.name}  (sequence name {sp.sequence_name})")
        if sp.error:
            lines.append(f"  ERROR: {sp.error}")
            continue
        lines.append(f"  frame rate : {sp.fps:g} fps  [{sp.fps_source}]")
        lines.append(f"  QPs        : {', '.join(str(q.qp) for q in sp.qps)}  [{sp.qp_source}]")
        lines.append(f"  annotations: {sp.annotations.folder}  types={sp.annotations.types()}"
                     + (f"  {sp.annotations.name}_masks/" if sp.annotations.masks_dir else ""))
        if isinstance(sp.classes, dict):
            lines.append("  classes    : "
                         + "; ".join(f"{t}: {', '.join(class_names(c))}" for t, c in sp.classes.items())
                         + "  (only these classes in those tasks)")
        elif sp.classes is not None:
            lines.append(f"  classes    : {', '.join(class_names(sp.classes))}  (only these in detection, "
                         f"instance segmentation and tracking; "
                         f"panoptic / semantic: these things + all stuff classes; not keypoints)")
        if sp.obb_classes is not None:
            from .annotations import dota_label_map
            names = {c: n for n, c in dota_label_map().items()}
            lines.append(f"  obb classes: {', '.join(names[c] for c in sp.obb_classes)}  "
                         f"(only these in oriented_detection)")
        for q in sp.qps:
            src = q.bitstream.name if q.bitstream else "-"
            lines.append(f"  QP{q.qp:<3} bitstream={src}  pngs={len(q.pngs)}"
                         f"  log={'yes' if q.log_file else 'no'}")
        for t in sp.tasks:
            models = ", ".join(model_id(m) for m in t.models)
            extra = f"  ({t.reason})" if t.reason else ""
            lines.append(f"  task {t.task:<22} {t.status:<8} models: {models}{extra}")
            for mid, why in t.skipped_models.items():
                lines.append(f"       not run: {mid}  ({why})")
        for w in sp.warnings:
            lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- execution

def run_sequence(sp: SequencePlan, cfg: dict, ffmpeg_path: str | None, ffmpeg_version: str | None,
                 pool=None, env: dict | None = None) -> dict:
    """Run all implemented stages for one sequence. Errors in one QP/model do not stop the others."""
    from .annotations import (
        AnnotationError,
        has_track_ids,
        load_bib_annotations,
        load_detection_annotations,
        load_lpr_annotations,
        load_obb_annotations,
        load_panoptic_annotations,
        load_pose_annotations,
    )
    from .decode import DecodeError, ensure_frames, remove_frames
    from .infer import (
        InferenceError,
        cached_predictions,
        depth_maps_present,
        predictions_path,
        remove_depth_maps,
        run_inference,
    )
    from .paths import read_encoder_log_info
    from .score import (
        cached_depth_scores,
        score_depth,
        score_detection,
        score_keypoints,
        score_lpr,
        score_obb,
        score_panoptic,
        score_segmentation,
        score_tracking,
        scores_path,
    )

    status: dict = {"sequence": sp.name, "qps": {}, "errors": []}
    det_ann = None
    if any(tp.status == "run" and tp.annotation_type == "detection" for tp in sp.tasks):
        try:
            det_ann = load_detection_annotations(sp.annotations.detection_json)
            log.info(f"Annotations: {sp.annotations.detection_json.name}  {len(det_ann.frames)} frames, "
                     f"classes {sorted(det_ann.class_ids)}")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot load detection annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "detection" and tp.status == "run":
                    tp.status, tp.reason = "error", f"annotations: {e}"
    pan_ann = None
    pan_masks = sp.annotations.panoptic_masks_dir if sp.annotations else None
    if any(tp.status == "run" and tp.annotation_type == "panoptic" for tp in sp.tasks):
        try:
            if pan_masks is None:
                raise AnnotationError(f"no {sp.annotations.name}_panoptic_masks/ folder")
            pan_ann = load_panoptic_annotations(sp.annotations.panoptic_json)
            log.info(f"Panoptic annotations: {sp.annotations.panoptic_json.name}  {len(pan_ann.frames)} frames, "
                     f"{len(pan_ann.class_ids)} classes")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot use panoptic annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "panoptic" and tp.status == "run":
                    tp.status, tp.reason = "error", f"panoptic annotations: {e}"
    depth_files, depth_sha, depth_min = None, None, sequence_depth_min_mm(cfg, sp.name)
    if any(tp.status == "run" and tp.annotation_type == "depth" for tp in sp.tasks):
        try:
            from .metrics.depth import gt_fingerprint
            from .paths import scan_pngs
            depth_files = scan_pngs(sp.annotations.depth_dir)
            if not depth_files:
                raise AnnotationError(f"no depth PNGs in {sp.annotations.depth_dir}")
            depth_sha = gt_fingerprint(depth_files)
            log.info(f"Depth ground truth: {sp.annotations.depth_dir.name}/  {len(depth_files)} frames; pixels with "
                     f"depth 0 or < {depth_min} mm ignored (depth_min_mm)")
        except (AnnotationError, DiscoveryError, OSError) as e:
            log.error(f"{sp.name}: cannot use depth ground truth: {e}")
            status["errors"].append(str(e))
            depth_files = None
            for tp in sp.tasks:
                if tp.annotation_type == "depth" and tp.status == "run":
                    tp.status, tp.reason = "error", f"depth ground truth: {e}"
    pose_ann = None
    if any(tp.status == "run" and tp.annotation_type == "pose" for tp in sp.tasks):
        try:
            pose_ann = load_pose_annotations(sp.annotations.pose_json)
            n = sum(len(v) for v in pose_ann.frames.values())
            log.info(f"Pose annotations: {sp.annotations.pose_json.name}  {len(pose_ann.frames)} frames, {n} persons")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot load pose annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "pose" and tp.status == "run":
                    tp.status, tp.reason = "error", f"pose annotations: {e}"
    obb_ann = None
    if any(tp.status == "run" and tp.annotation_type == "obb" for tp in sp.tasks):
        try:
            obb_ann = load_obb_annotations(sp.annotations.obb_json)
            n = sum(len(v) for v in obb_ann.frames.values())
            log.info(f"OBB annotations: {sp.annotations.obb_json.name}  {len(obb_ann.frames)} frames, {n} objects, "
                     f"DOTA classes {sorted(obb_ann.class_ids)}")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot load OBB annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "obb" and tp.status == "run":
                    tp.status, tp.reason = "error", f"OBB annotations: {e}"
    lpr_ann = None
    if any(tp.status == "run" and tp.annotation_type == "lpr" for tp in sp.tasks):
        try:
            lpr_ann = load_lpr_annotations(sp.annotations.lpr_json)
            n = sum(len(v) for v in lpr_ann.frames.values())
            log.info(f"Licence plate annotations: {sp.annotations.lpr_json.name}  {len(lpr_ann.frames)} frames, "
                     f"{n} plates")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot load licence plate annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "lpr" and tp.status == "run":
                    tp.status, tp.reason = "error", f"licence plate annotations: {e}"
    bib_ann = None
    if any(tp.status == "run" and tp.annotation_type == "bib" for tp in sp.tasks):
        try:
            bib_ann = load_bib_annotations(sp.annotations.bib_json)
            n = sum(len(v) for v in bib_ann.frames.values())
            log.info(f"Race bib annotations: {sp.annotations.bib_json.name}  {len(bib_ann.frames)} frames, {n} bibs")
        except (AnnotationError, OSError, ValueError) as e:
            log.error(f"{sp.name}: cannot load race bib annotations: {e}")
            status["errors"].append(str(e))
            for tp in sp.tasks:
                if tp.annotation_type == "bib" and tp.status == "run":
                    tp.status, tp.reason = "error", f"race bib annotations: {e}"
    masks_dir = sp.annotations.masks_dir if sp.annotations else None
    if det_ann is not None:
        from .metrics.coco_segm import segmentation_source
        for tp in sp.tasks:
            if tp.task == "tracking" and tp.status == "run" and not has_track_ids(det_ann):
                tp.status, tp.reason = "skipped", "no track_id in the annotations"
                log.warning(f"{sp.name}: tracking skipped: {tp.reason}")
            if tp.task == "instance_segmentation" and tp.status == "run":
                src = segmentation_source(det_ann, masks_dir)
                if src is None:
                    tp.status, tp.reason = "skipped", "no instance masks folder and no polygons in the annotations"
                    log.warning(f"{sp.name}: instance_segmentation skipped: {tp.reason}")
                else:
                    log.info(f"Instance masks: {src}" + (f" ({masks_dir.name}/)" if src == "masks" else ""))
    for q in sp.qps:
        qs: dict = {}
        status["qps"][q.qp] = qs
        try:
            expected = read_encoder_log_info(q.log_file).get("frames")
            qs["decode"] = ensure_frames(q, cfg, ffmpeg_path, ffmpeg_version, expected)
        except DecodeError as e:
            log.error(str(e))
            qs["error"] = str(e)
            status["errors"].append(str(e))
            continue
        if pool is None:
            continue
        decode_settings = qs["decode"].get("settings")
        for tp in sp.tasks:
            if tp.status != "run":
                continue
            for m in tp.models:
                mid = model_id(m)
                label = f"QP{q.qp} {tp.task}/{mid}"
                pred_path = predictions_path(sp.results_dir, mid, q.qp)
                if tp.task == "depth" and not depth_maps_present(pred_path):
                    # depth maps removed (inference.remove_depth_maps): still valid if predictions and scores are
                    sc_path = scores_path(sp.results_dir, mid, q.qp, tp.task)
                    sc = (cached_depth_scores(depth_files, pred_path, q.pngs, sc_path, label, depth_min, depth_sha)
                          if cached_predictions(pred_path, m, decode_settings, len(q.pngs), sp.fps,
                                                require_depth_maps=False) is not None else None)
                    if sc is not None:
                        qs.setdefault("inference", {})[mid] = "ok"
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                        continue
                    if pred_path.is_file():
                        log.info(f"{label}: depth maps missing and the scores need recomputing: running inference")
                try:
                    run_inference(pool, m, q.pngs, pred_path, decode_settings,
                                  cfg["inference"]["timeout_s"], label, env, tp.task, sp.fps)
                    qs.setdefault("inference", {})[mid] = "ok"
                except InferenceError as e:
                    log.error(str(e))
                    qs.setdefault("inference", {})[mid] = f"error: {e}"
                    status["errors"].append(str(e))
                    continue
                try:
                    if tp.task == "detection":
                        sc = score_detection(det_ann, pred_path, q.pngs,
                                             scores_path(sp.results_dir, mid, q.qp), label, sp.task_classes(tp.task))
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task == "tracking":
                        sc = score_tracking(det_ann, pred_path, q.pngs,
                                            scores_path(sp.results_dir, mid, q.qp, tp.task), label,
                                            sp.task_classes(tp.task))
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task == "instance_segmentation":
                        sc = score_segmentation(det_ann, masks_dir, pred_path, q.pngs,
                                                scores_path(sp.results_dir, mid, q.qp, tp.task), label,
                                                sp.results_dir / "cache", sp.task_classes(tp.task))
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task == "depth":
                        sc = score_depth(depth_files, pred_path, q.pngs,
                                         scores_path(sp.results_dir, mid, q.qp, tp.task), label, depth_min, depth_sha)
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                        if sequence_remove_depth_maps(cfg, sp.name):
                            n = remove_depth_maps(pred_path)
                            if n:
                                log.info(f"{label}: removed {n} depth map PNGs (remove_depth_maps)")
                    elif tp.task in ("plate_recognition", "bib_recognition"):
                        sc = score_lpr(lpr_ann if tp.task == "plate_recognition" else bib_ann, pred_path, q.pngs,
                                       scores_path(sp.results_dir, mid, q.qp, tp.task), label,
                                       sequence_lpr_iou(cfg, sp.name), tp.task)
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task == "oriented_detection":
                        sc = score_obb(obb_ann, pred_path, q.pngs, scores_path(sp.results_dir, mid, q.qp, tp.task),
                                       label, sp.obb_classes, sequence_obb_score(cfg, sp.name) == "vehicle")
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task == "keypoints":
                        sc = score_keypoints(pose_ann, pred_path, q.pngs,
                                             scores_path(sp.results_dir, mid, q.qp, tp.task), label)
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                    elif tp.task in ("panoptic_segmentation", "semantic_segmentation"):
                        sc = score_panoptic(pan_ann, pan_masks, pred_path, q.pngs,
                                            scores_path(sp.results_dir, mid, q.qp, tp.task), label,
                                            sp.results_dir / "cache", sp.task_classes(tp.task))
                        qs.setdefault("scores", {}).setdefault(tp.task, {})[mid] = sc["result"]["stats"]
                except AnnotationError as e:
                    log.error(str(e))
                    status["errors"].append(str(e))

    if cfg["decode"]["remove_frames"]:
        for q in sp.qps:
            n = remove_frames(q)
            if n:
                log.info(f"QP{q.qp}: removed {n} decoded PNGs (--remove-frames)")
    return status
