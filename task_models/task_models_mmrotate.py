#!/usr/bin/env python3
"""
MMRotate (OpenMMLab) oriented object detectors, from ai4rs (https://github.com/wokaikaixinxin/ai4rs, Apache-2.0;
an MMRotate 1.x fork). DOTA v1.0 models: O2-RTDETR (RT-DETR with rotated boxes, arXiv 2603.15497), Rotated
RTMDet (RTMDet-R, one-stage) and Oriented R-CNN (two-stage), the last two with the official OpenMMLab weights.

Runs in its own venv (.venv-mmrotate, envs/mmrotate/setup.sh): mmcv needs compiled CUDA ops and the only prebuilt
wheels are for torch 2.1 + CUDA 12.1. The main pipeline starts this file as a worker subprocess
(python task_models_mmrotate.py --worker) and exchanges length-prefixed messages over stdin/stdout, as
task_models_roboflow.py (stdout of the libraries is redirected to stderr). The ai4rs checkout (its mmrotate package
and projects/) is put on sys.path; its root is passed by the main process (envs/mmrotate/ai4rs).

Models (name = ai4rs config name; released DOTA v1.0 checkpoints, single-scale 1024 x 1024 split training;
DOTA-v1.0 test mAP50 as reported in the ai4rs READMEs):
  o2_rtdetr_r18vd_2xb4_72e_dota          77.31   ModelScope (ai4rs)
  o2_rtdetr_r34vd_2xb4_72e_dota          78.13
  o2_rtdetr_r50vd_2xb4_72e_dota          78.45
  rotated_rtmdet_{tiny,s,m,l}-3x-dota    75.36 / 76.93 / 78.24 / 78.85   download.openmmlab.com (MMRotate 1.x)
  oriented-rcnn-le90_r50_fpn_1x_dota     75.69   download.openmmlab.com (MMRotate 0.x checkpoint, as linked)

Inference = the config's test pipeline (mmdet.apis.inference_detector), in one of two modes (input_size):
  'tiles' (default) = the DOTA single-scale test protocol "(1024,1024,200)": the frame at full resolution is cut into
    1024 x 1024 tiles with 200 px overlap (step 824; the last tile aligned to the frame edge; 1920x1080 -> 3 x 2
    tiles, no padding), each tile run with resize factor 1 (smaller frames: padded as the config), then the tile
    boxes are shifted to frame coordinates and merged per class with rotated NMS at IoU 0.1 (as MMRotate's
    DOTAMetric merge_patches). Objects keep the scale of the DOTA training crops. A whole frame resized to 1024
    halves the cars (RTMDet-R found almost none) and the scores of RTMDet-R (global channel attention) depend on
    the amount of padding of a whole padded frame, so tiles are used.
  N: the whole frame resized to fit N x N (keep ratio; 1920x1080 -> 1024x576 for N = 1024, the released
    pipelines), padded as the config (O2-RTDETR: to N x N with 114).
BGR input; the config's test_cfg (O2-RTDETR: top 300, no score threshold; RTMDet-R / Oriented R-CNN: score 0.05,
rotated NMS 0.1, at most 2000). Boxes come back in frame pixels as
(cx, cy, w, h, angle in rad) and are converted to 4 corners.

Output per object: [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4] normalised to the frame, class_id = DOTA v1.0
0-based id of labels_dota.py (Ultralytics order; the MMRotate order differs and is mapped by class name).
"""

import hashlib
import json
import os
import struct
import sys

_MS = "https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/{0}/epoch_72.pth"
_OMM = "https://download.openmmlab.com/mmrotate/"
# name -> (config file relative to the ai4rs root, weights URL)
MODELS = {
    **{n: (f"projects/rotated_rtdetr/configs/{n}.py", _MS.format(n))
       for n in ("o2_rtdetr_r18vd_2xb4_72e_dota", "o2_rtdetr_r34vd_2xb4_72e_dota", "o2_rtdetr_r50vd_2xb4_72e_dota")},
    **{f"rotated_rtmdet_{s}-3x-dota": (f"configs/rotated_rtmdet/rotated_rtmdet_{s}-3x-dota.py",
                                        f"{_OMM}v1.0/rotated_rtmdet/rotated_rtmdet_{s}-3x-dota/"
                                        f"rotated_rtmdet_{s}-3x-dota-{h}.pth")
       for s, h in (("tiny", "9d821076"), ("s", "11f6ccf5"), ("m", "beeadda6"), ("l", "23992372"))},
    "oriented-rcnn-le90_r50_fpn_1x_dota": (
        "configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py",
        f"{_OMM}v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth"),
}


def _download(url: str, dst: str) -> None:
    import urllib.request
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst)


class MMRotateOBB:
    """Input: RGB uint8 numpy array (H, W, 3)."""

    obb = True

    def __init__(self, model_cfg_name, device="cuda:0", confidence_threshold=None, ai4rs_dir=None, cache_dir=None,
                 input_size="tiles"):
        if model_cfg_name not in MODELS:
            raise ValueError(f"unknown mmrotate model '{model_cfg_name}' (known: {sorted(MODELS)})")
        sys.path.insert(0, ai4rs_dir)
        from labels_dota import DOTA_label_map
        from mmdet.apis import init_detector
        from mmrotate.utils import register_all_modules
        register_all_modules()
        cfg_rel, url = MODELS[model_cfg_name]
        self.config_path = os.path.join(ai4rs_dir, cfg_rel)
        self.weights_url = url
        os.makedirs(cache_dir, exist_ok=True)
        ckpt = url.rsplit("/", 1)[1]
        self.weights_path = os.path.join(cache_dir, ckpt if ckpt != "epoch_72.pth" else f"{model_cfg_name}_{ckpt}")
        if not os.path.isfile(self.weights_path):
            _download(url, self.weights_path)
        from mmengine.config import Config
        cwd = os.getcwd()
        os.chdir(ai4rs_dir)                  # configs import projects.* relative to the repository root
        try:
            cfg = Config.fromfile(self.config_path)
            self._set_input_size(cfg, input_size)
            self.model = init_detector(cfg, self.weights_path, device=device)
        finally:
            os.chdir(cwd)
        self.confidence_threshold = confidence_threshold
        self.mode = input_size
        names = [str(c).replace("-", " ") for c in self.model.dataset_meta["classes"]]
        unknown = [n for n in names if n not in DOTA_label_map]
        if unknown or len(names) != len(DOTA_label_map):
            raise RuntimeError(f"{model_cfg_name}: classes differ from DOTA v1.0 (labels_dota.py): {unknown}")
        self.id_map = [DOTA_label_map[n] for n in names]
        pipe = self.model.cfg.test_dataloader.dataset.pipeline
        tc = self.model.cfg.model.test_cfg
        tc = tc.get("rcnn", tc)                     # two-stage: the R-CNN stage gives the final boxes
        self.input_size = {"input_size": input_size,
                           **({"tile": self.TILE, "tile_overlap": self.TILE_GAP, "merge": f"nms_rotated "
                               f"{self.MERGE_IOU} per class"} if input_size == "tiles" else {}),
                           "resize": [list(s["scale"]) if s.get("scale") else f"scale_factor {s.get('scale_factor')}"
                                      for s in pipe if "Resize" in str(s["type"])],
                           "pad": [list(s["size"]) if s.get("size") else f"size_divisor {s.get('size_divisor')}"
                                   for s in pipe if "Pad" in str(s["type"])],
                           "max_per_img": int(tc.get("max_per_img", 0)),
                           "test_score_thr": tc.get("score_thr"),
                           "test_nms": dict(tc["nms"]) if tc.get("nms") else None}

    TILE, TILE_GAP, MERGE_IOU = 1024, 200, 0.1

    @staticmethod
    def _set_input_size(cfg, input_size) -> None:
        pipe = cfg.test_dataloader.dataset.pipeline
        resize = [s for s in pipe if "Resize" in str(s["type"])]
        if len(resize) != 1:
            raise RuntimeError(f"expected one Resize in the test pipeline, found {len(resize)}")
        r = resize[0]
        if input_size == "tiles":
            r.pop("scale", None)                     # tiles at full resolution (Pad, if any, keeps 1024 x 1024)
            r["scale_factor"] = 1.0
            r["keep_ratio"] = True
        else:
            n = int(input_size)
            r["scale"] = (n, n)
            for p in pipe:
                if "Pad" in str(p["type"]) and p.get("size"):
                    p["size"] = (n, n)

    @classmethod
    def tile_starts(cls, length: int) -> list[int]:
        """Return the tile start offsets along one frame side (step TILE - TILE_GAP, last tile aligned to the end)."""
        step = cls.TILE - cls.TILE_GAP
        s = list(range(0, max(length - cls.TILE, 0) + 1, step))
        if s[-1] + cls.TILE < length:
            s.append(length - cls.TILE)
        return s

    def _infer_boxes(self, bgr):
        """(boxes [N, 5] cx, cy, w, h, angle in frame pixels, scores [N], labels [N]) on the GPU."""
        import numpy as np
        import torch
        from mmdet.apis import inference_detector
        if self.mode != "tiles":
            p = inference_detector(self.model, bgr).pred_instances
            return (p.bboxes.tensor if hasattr(p.bboxes, "tensor") else p.bboxes).float(), p.scores, p.labels
        from mmcv.ops import nms_rotated
        H, W = bgr.shape[:2]
        offs = [(x, y) for y in self.tile_starts(H) for x in self.tile_starts(W)]
        tiles = [np.ascontiguousarray(bgr[y:y + self.TILE, x:x + self.TILE]) for x, y in offs]
        res = inference_detector(self.model, tiles)
        B, S, L = [], [], []
        for (x, y), r in zip(offs, res):
            p = r.pred_instances
            b = (p.bboxes.tensor if hasattr(p.bboxes, "tensor") else p.bboxes).float().clone()
            b[:, 0] += x
            b[:, 1] += y
            B.append(b)
            S.append(p.scores.float())
            L.append(p.labels)
        B, S, L = torch.cat(B), torch.cat(S), torch.cat(L)
        if len(offs) == 1 or not len(S):
            return B, S, L
        keep = []
        for c in L.unique():
            idx = (L == c).nonzero().flatten()
            _, k = nms_rotated(B[idx], S[idx], self.MERGE_IOU)
            keep.append(idx[k])
        keep = torch.cat(keep)
        return B[keep], S[keep], L[keep]

    def forward(self, img_rgb):
        """Detect oriented objects in one RGB uint8 frame, a numpy array (H, W, 3).

        Returns:
            Per object [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4] (DOTA 0-based id, corners normalised), by
            descending score; objects below confidence_threshold (if set) are dropped.
        """
        import numpy as np
        import torch
        from mmrotate.structures.bbox import rbox2qbox
        H, W = img_rgb.shape[:2]
        bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
        with torch.no_grad():
            boxes, scores, labels = self._infer_boxes(bgr)
        q = rbox2qbox(boxes).cpu().numpy()                             # (N, 8) corners in pixels
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        out = []
        for i in np.argsort(-scores, kind="stable"):
            s = float(scores[i])
            if self.confidence_threshold is not None and s < self.confidence_threshold:
                continue
            c = q[i]
            out.append([self.id_map[int(labels[i])], s] +
                       [float(c[k] / (W if k % 2 == 0 else H)) for k in range(8)])
        return out


def state_dict_hash(module) -> str:
    """Same algorithm as vcmtest.infer.state_dict_hash (kept in sync; this venv has no vcmtest deps)."""
    import torch
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(f"{k}|{t.dtype}|{tuple(sd[k].shape)}|".encode())
        if t.numel():
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- worker protocol
# message = 4-byte big-endian length + JSON header [+ raw payload of header["nbytes"] bytes]

def _recv(f):
    n = f.read(4)
    if len(n) < 4:
        return None, None
    hdr = json.loads(f.read(struct.unpack(">I", n)[0]))
    payload = f.read(hdr["nbytes"]) if hdr.get("nbytes") else None
    return hdr, payload


def _send(f, obj):
    b = json.dumps(obj).encode()
    f.write(struct.pack(">I", len(b)) + b)
    f.flush()


def _git_commit(path: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def _worker():
    proto_in = sys.stdin.buffer
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                       # anything printed by libraries goes to stderr
    sys.stdout = sys.stderr
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    net = None
    while True:
        hdr, payload = _recv(proto_in)
        if hdr is None or hdr["cmd"] == "quit":
            return
        try:
            if hdr["cmd"] == "load":
                import random

                import numpy as np
                import torch
                random.seed(hdr.get("seed", 0))
                np.random.seed(hdr.get("seed", 0))
                torch.manual_seed(hdr.get("seed", 0))
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                net = MMRotateOBB(hdr["model"], hdr["device"], hdr.get("score_threshold"), hdr["ai4rs_dir"],
                                  hdr["cache_dir"], hdr.get("input_size", "tiles"))
                from importlib.metadata import version
                _send(proto_out, {"ok": True, "obb": True, "score_threshold": net.confidence_threshold,
                                  "weights_sha256": state_dict_hash(net.model),
                                  "input_size": net.input_size,
                                  "config": os.path.relpath(net.config_path, hdr["ai4rs_dir"]),
                                  "ai4rs_commit": _git_commit(hdr["ai4rs_dir"]),
                                  "versions": {p: version(p) for p in ("torch", "mmcv", "mmdet", "mmengine",
                                                                       "numpy")}})
            elif hdr["cmd"] == "predict":
                import numpy as np
                img = np.frombuffer(payload, dtype=np.uint8).reshape(hdr["shape"])
                _send(proto_out, {"ok": True, "preds": net.forward(img)})
            else:
                _send(proto_out, {"ok": False, "error": f"unknown command {hdr['cmd']}"})
        except Exception as e:  # reported to the main process, which decides what to do
            import torch
            _send(proto_out, {"ok": False, "error": f"{type(e).__name__}: {e}",
                              "oom": isinstance(e, torch.cuda.OutOfMemoryError)})


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        print(__doc__)
