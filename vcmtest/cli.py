"""Command line interface.

  python vcmtest [config.yaml] [--seq NAME ...] [--qp N ...] [--remove-frames] [--remove-depth-maps] [--plan]
  python vcmtest [config.yaml] --bd [--seq NAME ...]        BD comparisons only (configured anchor/test pairs)
  python vcmtest [config.yaml] --bd --anchor A --test T     BD comparison of one pair only

Without a config file, configs/config.yaml is used.
"""

from __future__ import annotations

import argparse
import logging

from .bd import configured_pairs, run_bd
from .config import WORKER_PYTHON, ConfigError, load_config, missing_model_requirements, resolve_path
from .env_info import add_file_log, environment_info, remove_handler, set_seeds, setup_logging, timestamp
from .paths import check_ffmpeg, probe_ffmpeg, resolve_ffmpeg
from .pipeline import build_plan, describe, run_sequence
from .results import write_sequence_results, write_summary

log = logging.getLogger("vcmtest")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    p = argparse.ArgumentParser(prog="vcmtest", description="VCM rate-accuracy evaluation testbench")
    p.add_argument("config", nargs="?", default=None, help="YAML config (default: configs/config.yaml)")
    p.add_argument("--seq", nargs="+", metavar="NAME", help="sequence folder name(s); overrides config")
    p.add_argument("--qp", nargs="+", type=int, metavar="N", help="QP(s); overrides config")
    p.add_argument("--remove-frames", action="store_true", help="delete decoded PNGs after evaluation")
    p.add_argument("--remove-depth-maps", action="store_true",
                   help="delete predicted depth map PNGs once scored "
                        "(config inference.remove_depth_maps, also per sequence)")
    p.add_argument("--plan", action="store_true", help="only discover inputs and print the work plan")
    p.add_argument("--bd", action="store_true",
                   help="only the BD comparisons (from the results on disk): the configured anchor/test pairs "
                        "(configs/anchors.yaml, bd.anchors; with --seq: the pairs involving those folders), "
                        "or the one pair given with --anchor/--test")
    p.add_argument("--anchor", metavar="FOLDER", help="anchor folder of a single BD comparison (with --test)")
    p.add_argument("--test", metavar="FOLDER", help="test folder of a single BD comparison (with --anchor)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Command-line entry point: run the configured pipeline (or only the BD comparisons with --bd / --anchor --test).

    Returns:
        Exit code: 0 = success, 1 = errors in a sequence (or in the plan with --plan), 2 = configuration error.
    """
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        log.error(str(e))
        return 2
    if bool(args.anchor) != bool(args.test):
        log.error("--anchor and --test go together")
        return 2
    if args.anchor:
        args.bd = True
    if args.bd:
        return _bd_only(cfg, args)
    if args.remove_frames:
        cfg["decode"]["remove_frames"] = True
    if args.remove_depth_maps:
        cfg["_cli_remove_depth_maps"] = True

    run_log = add_file_log(resolve_path(cfg["results_root"]) / "logs" / f"run_{timestamp()}.log")
    try:
        log.info(f"Config: {cfg['_config_path']}")
        _log_anchors(cfg)
        set_seeds(cfg["seed"])
        plans = build_plan(cfg, args.seq, args.qp)

        ffmpeg_line, ff = None, None
        ok_plans = [sp for sp in plans if not sp.error]
        if any(q.bitstream for sp in ok_plans for q in sp.qps):
            try:
                ff = probe_ffmpeg(resolve_ffmpeg(cfg))
                for w in check_ffmpeg(ff, {"vvc", "hevc"}):
                    log.warning(w)
                ffmpeg_line = ff.version_line
                log.info(f"ffmpeg: {ff.path}  [{ff.version_line}]")
                if cfg["decode"]["decoder"] != "ffmpeg" and any(sp.needs_decode for sp in ok_plans):
                    from .decoders import get_decoder
                    dec = get_decoder(cfg)                    # fails here if the executable is missing
                    log.info(f"decoder: {dec.name} {getattr(dec, 'executable', '')}  [{dec.version}]")
            except ConfigError as e:
                if any(sp.needs_decode for sp in ok_plans):
                    raise
                log.warning(f"{e} (not needed now: all frames already exist)")

        env = environment_info(ffmpeg_line)
        log.info(f"Environment: torch {env['packages']['torch']}, CUDA {env.get('cuda')}, "
                 f"GPU {env.get('gpu')}, git {env['git_commit']}")
        log.info("Work plan:\n" + describe(plans))

        n_err = sum(1 for sp in plans if sp.error)
        missing = missing_model_requirements(cfg)
        for msg in missing:
            log.error(f"Missing model requirement: {msg}")
        if args.plan:
            return 1 if n_err or missing else 0
        if missing:
            raise ConfigError("models in the config need packages that are not installed (see above)")

        ffmpeg_path = ff.path if ffmpeg_line else None
        if str(cfg["device"]).startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise ConfigError(f"device '{cfg['device']}' requested but CUDA is not available "
                                  f"(check the container's GPU access, or set device: cpu)")
        from .infer import ModelPool, setup_weight_dirs
        setup_weight_dirs(resolve_path(cfg["weights_dir"]))
        pool = ModelPool(cfg["device"], resolve_path(cfg["weights_dir"]), cfg["seed"],
                         worker_python={k: resolve_path(cfg[k]) for k in sorted(set(WORKER_PYTHON.values()))})
        for sp in plans:
            if sp.error:
                continue
            seq_log = add_file_log(sp.results_dir / "logs" / f"{timestamp()}.log")
            try:
                log.info(f"=== {sp.name}: {len(sp.qps)} QP(s)")
                for w in sp.warnings:
                    log.warning(w)
                st = run_sequence(sp, cfg, ffmpeg_path, ffmpeg_line, pool, env)
                n_err += bool(st["errors"])
                write_sequence_results(sp, cfg, st, env)
            finally:
                remove_handler(seq_log)
        write_summary(cfg)
        if cfg["bd"]["enabled"]:
            run_bd(cfg, configured_pairs(cfg, [sp.name for sp in plans if not sp.error]))
        log.info("Done" + (f" with errors in {n_err} sequence(s)" if n_err else ""))
        return 1 if n_err else 0
    except ConfigError as e:
        log.error(str(e))
        return 2
    finally:
        remove_handler(run_log)


def _log_anchors(cfg: dict) -> None:
    for w in cfg.get("_anchor_warnings", []):
        log.warning(w)
    for seq, inh in sorted(cfg.get("_inherited_overrides", {}).items()):
        log.info(f"{seq}: sequence_overrides {inh['keys']} inherited from its BD anchor {inh['anchor']}")


def _bd_only(cfg: dict, args) -> int:
    """--bd: BD comparisons from the results JSONs on disk; no decoding, inference or scoring."""
    run_log = add_file_log(resolve_path(cfg["results_root"]) / "logs" / f"bd_{timestamp()}.log")
    try:
        log.info(f"Config: {cfg['_config_path']}")
        _log_anchors(cfg)
        if args.anchor:
            data_root = resolve_path(cfg["data_root"])
            for f in (args.anchor, args.test):
                if not (data_root / f).is_dir():
                    raise ConfigError(f"no folder {f} in {data_root}")
            if args.anchor == args.test:
                log.warning("anchor = test: every BD value should be 0 (self-check)")
            elif args.anchor.split("_", 1)[0] != args.test.split("_", 1)[0]:
                log.warning(f"anchor and test have different sequence names ({args.anchor}, {args.test})")
            return 1 if run_bd(cfg, [(args.anchor, args.test)], strict=True) else 0
        pairs = configured_pairs(cfg, args.seq)
        if not pairs:
            log.warning("no anchor/test pairs (configs/anchors.yaml, bd.anchors)"
                        + (f" involving {args.seq}" if args.seq else ""))
        return 1 if run_bd(cfg, pairs) else 0
    except ConfigError as e:
        log.error(str(e))
        return 2
    finally:
        remove_handler(run_log)
