#!/usr/bin/env python3
"""
Route-1 multi-gene explicit transcription-chain model with stiffness-aware coarse-graining.

Design:
1) Keep high-granularity transcription chains (gene-specific explicit tx states).
2) Keep gene-specific translation/mRNA/protein channels.
3) Coarse-grain only stiff fast loops via configurable scales:
   - tx_fast_scale: scale explicit tx-chain fast steps uniformly.
   - ribo_bind_scale: scale ribosome bind/unbind pair uniformly (Kd preserved).
4) Optional extra coarse-grain on transcript chain length via tx-compress-scale.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Avoid matplotlib cache warnings in restricted home directories.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import simulate_route1_multigene_background as route1
import simulate_route1_twogene_explicit_tx_lines as etx


DEFAULT_OUTDIR = Path("output_route1_multigene_explicit_tx_hybrid")
DEFAULT_PANEL_TEMPLATE = Path("multigene_explicit_tx_gene_panel_template.json")
_LOCK_FD: Optional[int] = None


def _acquire_singleton_lock(enabled: bool, lock_key: str) -> Optional[Path]:
    """Prevent accidental multi-process contention for the same heavy simulation."""
    global _LOCK_FD
    if not bool(enabled):
        return None

    digest = hashlib.sha1(str(lock_key).encode("utf-8")).hexdigest()[:16]
    lock_path = Path("/tmp") / f"pure_multigene_sim_{digest}.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.lseek(fd, 0, os.SEEK_SET)
        holder = os.read(fd, 4096).decode("utf-8", errors="ignore").strip()
        os.close(fd)
        raise RuntimeError(
            "Another simulate_route1_multigene_explicit_tx_hybrid run is already active "
            f"(lock: {lock_path}). Holder: {holder or 'unknown'}"
        )

    holder_info = {
        "pid": int(os.getpid()),
        "started_at_unix_s": float(time.time()),
        "cwd": str(Path.cwd()),
        "cmd": " ".join(str(x) for x in os.sys.argv),
    }
    payload = json.dumps(holder_info, ensure_ascii=False)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload.encode("utf-8"))
    os.fsync(fd)
    _LOCK_FD = fd

    def _cleanup() -> None:
        global _LOCK_FD
        if _LOCK_FD is not None:
            try:
                os.close(_LOCK_FD)
            except Exception:
                pass
            _LOCK_FD = None

    atexit.register(_cleanup)
    return lock_path


@dataclass(frozen=True)
class GeneInput:
    name: str
    aa_length: int
    tl_scale: float = 1.0
    tx_scale: float = 1.0
    cds_dna: Optional[str] = None
    tx_nt_steps: Optional[int] = None
    dna_M: Optional[float] = None
    dna_weight: float = 1.0


def _default_six_gene_panel() -> List[GeneInput]:
    seeds: List[Tuple[str, int]] = [
        ("Pgi", 549),
        ("PfkA", 320),
        ("TpiA", 255),
        ("GAPDH", 331),
        ("Pgk", 387),
        ("Eno", 432),
    ]
    out: List[GeneInput] = []
    for name, aa_len in seeds:
        out.append(
            GeneInput(
                name=name,
                aa_length=int(aa_len),
                tl_scale=1.0,
                tx_scale=1.0,
                cds_dna=etx.get_known_cds_dna(name),
                tx_nt_steps=None,
                dna_M=None,
                dna_weight=1.0,
            )
        )
    return out


def _to_gene_input(obj: Dict[str, object]) -> GeneInput:
    if "name" not in obj:
        raise ValueError("gene entry missing 'name'")
    if "aa_length" not in obj:
        raise ValueError(f"gene '{obj.get('name')}' missing 'aa_length'")
    return GeneInput(
        name=str(obj["name"]),
        aa_length=int(max(1, int(obj["aa_length"]))),
        tl_scale=float(max(0.0, float(obj.get("tl_scale", 1.0)))),
        tx_scale=float(max(0.0, float(obj.get("tx_scale", 1.0)))),
        cds_dna=(None if obj.get("cds_dna", None) is None else str(obj.get("cds_dna"))),
        tx_nt_steps=(
            None
            if obj.get("tx_nt_steps", None) is None
            else int(max(1, int(obj.get("tx_nt_steps"))))
        ),
        dna_M=(None if obj.get("dna_M", None) is None else float(max(0.0, float(obj.get("dna_M"))))),
        dna_weight=float(max(0.0, float(obj.get("dna_weight", 1.0)))),
    )


def _write_default_template(path: Path) -> None:
    payload = {
        "dna_total_M": 5.0e-9,
        "genes": [
            {
                "name": g.name,
                "aa_length": int(g.aa_length),
                "tl_scale": float(g.tl_scale),
                "tx_scale": float(g.tx_scale),
                "cds_dna": g.cds_dna,
                "tx_nt_steps": g.tx_nt_steps,
                "dna_M": g.dna_M,
                "dna_weight": float(g.dna_weight),
            }
            for g in _default_six_gene_panel()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_gene_panel(gene_json: Optional[Path]) -> Tuple[List[GeneInput], Optional[float]]:
    if gene_json is None:
        return _default_six_gene_panel(), None

    payload = json.loads(gene_json.read_text())
    top_dna_total_M: Optional[float] = None
    if isinstance(payload, dict) and payload.get("dna_total_M", None) is not None:
        top_dna_total_M = float(max(0.0, float(payload.get("dna_total_M"))))

    if isinstance(payload, dict):
        raw_genes = payload.get("genes", [])
    elif isinstance(payload, list):
        raw_genes = payload
    else:
        raise ValueError("gene panel json must be a list or {'genes': [...]} object")

    genes = [_to_gene_input(obj) for obj in raw_genes]
    if len(genes) == 0:
        raise ValueError("gene panel is empty")
    return genes, top_dna_total_M


def _autofill_known_cds(genes: List[GeneInput]) -> List[GeneInput]:
    out: List[GeneInput] = []
    for g in genes:
        if g.cds_dna is not None and len(etx._clean_dna(g.cds_dna)) > 0:
            out.append(g)
            continue
        cds = etx.get_known_cds_dna(g.name)
        if cds is not None and len(etx._clean_dna(cds)) > 0:
            out.append(replace(g, cds_dna=cds))
        else:
            out.append(g)
    return out


def _apply_tx_compression(genes: List[GeneInput], tx_compress_scale: float) -> List[GeneInput]:
    s = float(max(1e-6, tx_compress_scale))
    tx_mode = str(os.environ.get("PURE_TX_PATHWAY_MODE", "")).strip().lower()
    use_tx1nt_baseline = tx_mode in {"1nt", "tx1nt", "finest_1nt"}
    out: List[GeneInput] = []
    for g in genes:
        if g.tx_nt_steps is not None:
            out.append(g)
            continue
        cds = etx._clean_dna(g.cds_dna or "")
        if len(cds) <= 0:
            out.append(g)
            continue
        if use_tx1nt_baseline:
            # Fine-profile compatible behavior: 1 nt baseline, then optional user compression.
            base_steps = float(len(cds))
        else:
            # Legacy behavior used in older multigene stiffness-mitigation runs.
            base_steps = float(len(cds)) * float(etx.DEFAULT_TX_COMPRESS_SCALE)
        steps = int(round(base_steps * s))
        steps = max(1, min(int(steps), int(len(cds))))
        out.append(replace(g, tx_nt_steps=steps))
    return out


def _allocate_gene_dna(
    genes: List[GeneInput],
    active_total_M: float,
) -> Dict[str, float]:
    total = float(max(0.0, active_total_M))
    fixed: Dict[str, float] = {}
    unspecified: List[GeneInput] = []
    for g in genes:
        if g.dna_M is None:
            unspecified.append(g)
        else:
            fixed[g.name] = float(max(0.0, float(g.dna_M)))

    fixed_sum = float(sum(fixed.values()))
    out: Dict[str, float] = {g.name: 0.0 for g in genes}

    if len(unspecified) == 0:
        if fixed_sum <= 0.0:
            share = total / float(max(1, len(genes)))
            return {g.name: float(max(0.0, share)) for g in genes}
        scale = total / fixed_sum
        for g in genes:
            out[g.name] = float(max(0.0, fixed.get(g.name, 0.0) * scale))
        return out

    rem = max(0.0, total - fixed_sum)
    for g in genes:
        if g.name in fixed:
            out[g.name] = float(max(0.0, fixed[g.name]))

    wsum = float(sum(float(max(0.0, g.dna_weight)) for g in unspecified))
    if wsum <= 0.0:
        share = rem / float(len(unspecified))
        for g in unspecified:
            out[g.name] = float(max(0.0, share))
    else:
        for g in unspecified:
            w = float(max(0.0, g.dna_weight))
            out[g.name] = float(max(0.0, rem * (w / wsum)))
    return out


def _single_control_dna(
    gene_names: List[str],
    dna_mux: Dict[str, float],
    target_gene: str,
    mode: str,
) -> Dict[str, float]:
    total = float(sum(float(max(0.0, dna_mux.get(g, 0.0))) for g in gene_names))
    out = {g: 0.0 for g in gene_names}
    if mode == "same_total":
        out[target_gene] = total
    else:
        out[target_gene] = float(max(0.0, dna_mux.get(target_gene, 0.0)))
    return out


def _build_gene_cfgs(genes: List[GeneInput]) -> List[etx.GeneConfig]:
    return [
        etx.GeneConfig(
            name=str(g.name),
            aa_length=int(max(1, g.aa_length)),
            tl_scale=float(max(0.0, g.tl_scale)),
            tx_scale=float(max(0.0, g.tx_scale)),
            cds_dna=(None if g.cds_dna is None else str(g.cds_dna)),
            tx_nt_steps=(None if g.tx_nt_steps is None else int(max(1, g.tx_nt_steps))),
        )
        for g in genes
    ]


def _write_live_status(path: Optional[Path], payload: Dict[str, object]) -> None:
    if path is None:
        return
    etx._write_status_json(path, payload)


def _safe_file_fingerprint(path: Path) -> Dict[str, object]:
    p = Path(path)
    try:
        st = p.stat()
        return {
            "path": str(p),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except Exception:
        return {"path": str(p), "size": -1, "mtime_ns": -1}


def _gene_signature_for_cache(genes: List[GeneInput]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for g in genes:
        cds = etx._clean_dna(g.cds_dna or "")
        out.append(
            {
                "name": str(g.name),
                "aa_length": int(g.aa_length),
                "tl_scale": float(g.tl_scale),
                "tx_scale": float(g.tx_scale),
                "tx_nt_steps": (None if g.tx_nt_steps is None else int(g.tx_nt_steps)),
                "dna_M": (None if g.dna_M is None else float(g.dna_M)),
                "dna_weight": float(g.dna_weight),
                "cds_len": int(len(cds)),
                "cds_sha1": hashlib.sha1(cds.encode("utf-8")).hexdigest() if len(cds) > 0 else "",
            }
        )
    return out


def _model_cache_key(
    *,
    args,
    genes: List[GeneInput],
    panel_dna_total: Optional[float],
) -> str:
    payload = {
        "schema": "multigene_explicit_tx_hybrid_model_cache_v1",
        "summary_json": _safe_file_fingerprint(Path(args.summary_json)),
        "hf_summary_json": _safe_file_fingerprint(Path(args.hf_summary_json)),
        "gene_config_json": (None if args.gene_config_json is None else _safe_file_fingerprint(Path(args.gene_config_json))),
        "force_fine_node_env": bool(int(args.force_fine_node_env)),
        "gamma_h": float(args.gamma_h),
        "fit_points": int(args.fit_points),
        "native_degfp_share": float(args.native_degfp_share),
        "tx_compress_scale": float(args.tx_compress_scale),
        "tx_fast_scale": float(args.tx_fast_scale),
        "ribo_bind_scale": float(args.ribo_bind_scale),
        "mrna_decay_base": float(args.mrna_decay_base),
        "mrna_bound_decay_ratio": float(args.mrna_bound_decay_ratio),
        "autofill_known_cds": bool(int(args.autofill_known_cds)),
        "dna_total_M": float(max(0.0, panel_dna_total if panel_dna_total is not None else args.dna_total_M)),
        "tx_mode_env": str(os.environ.get("PURE_TX_PATHWAY_MODE", "")),
        "tl_mode_env": str(os.environ.get("PURE_TL_MODE", "")),
        "genes": _gene_signature_for_cache(genes),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_bundle_paths(cache_dir: Path, model_key: str) -> Tuple[Path, Path]:
    root = cache_dir / model_key
    return root / "bundle.json", root / "model.xml"


def _load_model_cache(cache_dir: Path, model_key: str) -> Optional[Dict[str, object]]:
    bundle_path, sbml_path = _cache_bundle_paths(cache_dir, model_key)
    if (not bundle_path.exists()) or (not sbml_path.exists()):
        return None
    try:
        bundle = json.loads(bundle_path.read_text())
        if str(bundle.get("schema", "")) != "multigene_explicit_tx_hybrid_model_cache_v1":
            return None
        if str(bundle.get("model_key", "")) != str(model_key):
            return None
        return {
            "sbml_ext": sbml_path.read_text(),
            "counts_bg": dict(bundle.get("counts_bg", {}) or {}),
            "hf_meta": dict(bundle.get("hf_meta", {}) or {}),
            "tx_seq_meta": list(bundle.get("tx_seq_meta", []) or []),
            "aa_profile_meta": list(bundle.get("aa_profile_meta", []) or []),
            "protein_by_gene": dict(bundle.get("protein_by_gene", {}) or {}),
            "dna_by_gene": dict(bundle.get("dna_by_gene", {}) or {}),
            "delta": dict(bundle.get("delta", {}) or {}),
            "explicit_counts_now": dict(bundle.get("explicit_counts_now", {}) or {}),
            "bundle_path": str(bundle_path),
            "sbml_path": str(sbml_path),
        }
    except Exception:
        return None


def _save_model_cache(
    *,
    cache_dir: Path,
    model_key: str,
    sbml_ext: str,
    counts_bg: Dict[str, object],
    hf_meta: Dict[str, object],
    tx_seq_meta: List[Dict[str, object]],
    aa_profile_meta: List[Dict[str, object]],
    protein_by_gene: Dict[str, str],
    dna_by_gene: Dict[str, str],
    delta: Dict[str, object],
    explicit_counts_now: Dict[str, object],
) -> Tuple[Path, Path]:
    bundle_path, sbml_path = _cache_bundle_paths(cache_dir, model_key)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    sbml_path.write_text(str(sbml_ext))
    bundle = {
        "schema": "multigene_explicit_tx_hybrid_model_cache_v1",
        "model_key": str(model_key),
        "sbml_file": str(sbml_path),
        "counts_bg": dict(counts_bg),
        "hf_meta": dict(hf_meta),
        "tx_seq_meta": list(tx_seq_meta),
        "aa_profile_meta": list(aa_profile_meta),
        "protein_by_gene": {str(k): str(v) for k, v in dict(protein_by_gene).items()},
        "dna_by_gene": {str(k): str(v) for k, v in dict(dna_by_gene).items()},
        "delta": dict(delta),
        "explicit_counts_now": dict(explicit_counts_now),
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    return bundle_path, sbml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-gene explicit TX model with stiffness-aware coarse-graining.")
    parser.add_argument("--summary-json", type=Path, default=route1.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--hf-summary-json", type=Path, default=route1.DEFAULT_HF_SUMMARY)
    parser.add_argument("--force-fine-node-env", type=int, default=1)
    parser.add_argument("--gene-config-json", type=Path, default=None, help="Gene panel JSON (list or {'genes': [...]})")
    parser.add_argument("--write-template-json", type=Path, default=None, help="Write default template panel and exit")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gamma-h", type=float, default=1.0)
    parser.add_argument("--fit-points", type=int, default=120)
    parser.add_argument("--t-end-h", type=float, default=8.0)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument(
        "--progress-chunk-h",
        type=float,
        default=0.0,
        help="If >0, run segmented simulation and write live progress/checkpoints each chunk.",
    )
    parser.add_argument(
        "--progress-points-per-chunk",
        type=int,
        default=0,
        help="Optional sampling points per chunk for segmented run (0=auto).",
    )

    parser.add_argument("--native-degfp-share", type=float, default=0.0)
    parser.add_argument("--dna-total-M", type=float, default=5.0e-9, help="Total DNA across native + all explicit gene lines")
    parser.add_argument("--autofill-known-cds", type=int, default=1, help="Autofill known MG1655 CDS when missing (1/0)")

    parser.add_argument(
        "--tx-compress-scale",
        type=float,
        default=1.0,
        help="Extra compression on explicit tx chain length (1.0 = baseline CDS-length compression)",
    )
    parser.add_argument(
        "--tx-fast-scale",
        type=float,
        default=1.0,
        help="Uniform scale on explicit tx-chain fast steps (init/elong/release)",
    )
    parser.add_argument(
        "--ribo-bind-scale",
        type=float,
        default=1.0,
        help="Uniform scale on ribosome bind/unbind pair; lower values reduce stiffness while preserving Kd",
    )
    parser.add_argument("--mrna-decay-base", type=float, default=etx.DEFAULT_MRNA_DECAY_BASE)
    parser.add_argument("--mrna-bound-decay-ratio", type=float, default=etx.DEFAULT_MRNA_BOUND_DECAY_RATIO)
    parser.add_argument("--build-only", type=int, default=0, help="Only build explicit network and report counts (1/0)")
    parser.add_argument(
        "--singleton-lock",
        type=int,
        default=1,
        help="1 to enforce a single active run of this workflow on the machine (recommended)",
    )
    parser.add_argument(
        "--use-model-cache",
        type=int,
        default=1,
        help="1 to reuse cached SBML/network bundle when model settings are unchanged",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(".cache_multigene_explicit_tx_hybrid"),
        help="Directory for cached multigene model bundles",
    )
    parser.add_argument(
        "--rr-reuse-across-cases",
        type=int,
        default=1,
        help="1 to compile RR once and reuse across multiplex/single-control cases",
    )

    parser.add_argument("--run-single-controls", type=int, default=0)
    parser.add_argument("--single-dna-mode", type=str, choices=["same_total", "same_gene"], default="same_total")
    args = parser.parse_args()

    lock_path = _acquire_singleton_lock(
        enabled=bool(int(args.singleton_lock)),
        lock_key="route1_multigene_explicit_tx_hybrid_global",
    )
    if lock_path is not None:
        print(f"[lock] acquired singleton lock: {lock_path}", flush=True)

    if args.write_template_json is not None:
        _write_default_template(Path(args.write_template_json))
        print(Path(args.write_template_json))
        return

    genes, panel_dna_total = _load_gene_panel(args.gene_config_json)
    if bool(int(args.autofill_known_cds)):
        genes = _autofill_known_cds(genes)
    genes = _apply_tx_compression(genes, tx_compress_scale=float(args.tx_compress_scale))
    gene_cfgs = _build_gene_cfgs(genes)
    gene_names = [g.name for g in gene_cfgs]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress_enabled = float(args.progress_chunk_h) > 0.0
    progress_status = outdir / "multigene_explicit_tx_hybrid_progress_status.json" if progress_enabled else None
    progress_live_csv = outdir / "multigene_explicit_tx_hybrid_live.csv" if progress_enabled else None
    progress_ckpt_csv = outdir / "multigene_explicit_tx_hybrid_checkpoints.csv" if progress_enabled else None

    _write_live_status(
        progress_status,
        {
            "stage": "init",
            "message": "multigene simulation initialized",
            "progress_mode": {
                "enabled": bool(progress_enabled),
                "chunk_h": float(args.progress_chunk_h),
                "points_per_chunk": int(args.progress_points_per_chunk),
            },
        },
    )

    batch_params = etx._load_formal_batch_params(Path(args.summary_json))
    model_key = _model_cache_key(args=args, genes=genes, panel_dna_total=panel_dna_total)
    cache_dir = Path(args.model_cache_dir)
    cache_hit = False
    cache_bundle_path: Optional[Path] = None
    cache_sbml_path: Optional[Path] = None

    sbml_ext: str
    counts_bg: Dict[str, object]
    hf_meta: Dict[str, object]
    tx_seq_meta: List[Dict[str, object]]
    aa_profile_meta: List[Dict[str, object]]
    protein_by_gene: Dict[str, str]
    dna_by_gene: Dict[str, str]
    delta: Dict[str, object]
    explicit_counts_now: Dict[str, object]

    cached_bundle = (
        _load_model_cache(cache_dir=cache_dir, model_key=model_key)
        if bool(int(args.use_model_cache))
        else None
    )
    if cached_bundle is not None:
        cache_hit = True
        sbml_ext = str(cached_bundle["sbml_ext"])
        counts_bg = dict(cached_bundle.get("counts_bg", {}) or {})
        hf_meta = dict(cached_bundle.get("hf_meta", {}) or {})
        tx_seq_meta = list(cached_bundle.get("tx_seq_meta", []) or [])
        aa_profile_meta = list(cached_bundle.get("aa_profile_meta", []) or [])
        protein_by_gene = {str(k): str(v) for k, v in dict(cached_bundle.get("protein_by_gene", {}) or {}).items()}
        dna_by_gene = {str(k): str(v) for k, v in dict(cached_bundle.get("dna_by_gene", {}) or {}).items()}
        delta = dict(cached_bundle.get("delta", {}) or {})
        explicit_counts_now = dict(cached_bundle.get("explicit_counts_now", {}) or {})
        cache_bundle_path = Path(str(cached_bundle.get("bundle_path", "")))
        cache_sbml_path = Path(str(cached_bundle.get("sbml_path", "")))
        _write_live_status(
            progress_status,
            {
                "stage": "model_cache_hit",
                "message": "loaded cached explicit multigene model",
                "model_cache_key": str(model_key),
                "cache_bundle": (None if cache_bundle_path is None else str(cache_bundle_path)),
            },
        )
    else:
        sbml_bg, counts_bg, hf_meta = route1._rebuild_background_batch_model(
            gamma_h=float(args.gamma_h),
            fit_points=int(args.fit_points),
            hf_summary_json=Path(args.hf_summary_json),
            force_fine_node_env=bool(int(args.force_fine_node_env)),
        )
        _write_live_status(
            progress_status,
            {
                "stage": "background_ready",
                "message": "background model built",
                "counts_background": counts_bg,
            },
        )

        bg_doc = etx.libsbml.readSBMLFromString(sbml_bg)
        bg_model = bg_doc.getModel()
        if bg_model is None:
            raise RuntimeError("Invalid background SBML")

        tx_seq_meta = []
        aa_profile_meta = []
        for g in gene_cfgs:
            _, tx_meta = etx._build_gene_tx_rna_sequence(bg_model, g)
            tx_seq_meta.append(tx_meta)
            _, aa_meta = etx._build_gene_aa20_profile(g)
            aa_profile_meta.append(aa_meta)

        sbml_ext, protein_by_gene, dna_by_gene, delta = etx._extend_with_explicit_two_lines(
            sbml_text=sbml_bg,
            genes=gene_cfgs,
            native_degfp_share=float(args.native_degfp_share),
            mrna_decay_base=float(args.mrna_decay_base),
            mrna_bound_decay_ratio=float(args.mrna_bound_decay_ratio),
            tx_fast_scale=float(args.tx_fast_scale),
            ribo_bind_scale=float(args.ribo_bind_scale),
        )
        _write_live_status(
            progress_status,
            {
                "stage": "network_ready",
                "message": "explicit multigene network ready",
                "delta": delta,
                "n_genes": int(len(gene_cfgs)),
            },
        )

        sbml_doc = etx.libsbml.readSBMLFromString(sbml_ext)
        sbml_model = sbml_doc.getModel() if sbml_doc is not None else None
        explicit_counts_now = {
            "species_total": (None if sbml_model is None else int(sbml_model.getNumSpecies())),
            "reactions_total": (None if sbml_model is None else int(sbml_model.getNumReactions())),
        }

        if bool(int(args.use_model_cache)):
            cache_bundle_path, cache_sbml_path = _save_model_cache(
                cache_dir=cache_dir,
                model_key=model_key,
                sbml_ext=sbml_ext,
                counts_bg=counts_bg,
                hf_meta=hf_meta,
                tx_seq_meta=tx_seq_meta,
                aa_profile_meta=aa_profile_meta,
                protein_by_gene=protein_by_gene,
                dna_by_gene=dna_by_gene,
                delta=delta,
                explicit_counts_now=explicit_counts_now,
            )
            _write_live_status(
                progress_status,
                {
                    "stage": "model_cache_saved",
                    "message": "saved explicit multigene model cache",
                    "model_cache_key": str(model_key),
                    "cache_bundle": str(cache_bundle_path),
                },
            )

    dna_total_M = float(max(0.0, panel_dna_total if panel_dna_total is not None else args.dna_total_M))
    native_share = float(np.clip(float(args.native_degfp_share), 0.0, 1.0))
    native_dna = dna_total_M * native_share
    active_dna = max(0.0, dna_total_M - native_dna)
    dna_mux_by_name = _allocate_gene_dna(genes, active_total_M=active_dna)

    dna_mux = {"DNA": float(native_dna)}
    for g in gene_cfgs:
        sid = dna_by_gene[g.name]
        dna_mux[sid] = float(max(0.0, dna_mux_by_name.get(g.name, 0.0)))

    if ("species_total" not in explicit_counts_now) or ("reactions_total" not in explicit_counts_now):
        sbml_doc = etx.libsbml.readSBMLFromString(sbml_ext)
        sbml_model = sbml_doc.getModel() if sbml_doc is not None else None
        explicit_counts_now = {
            "species_total": (None if sbml_model is None else int(sbml_model.getNumSpecies())),
            "reactions_total": (None if sbml_model is None else int(sbml_model.getNumReactions())),
        }

    m_mux: Dict[str, Dict[str, float]] = {}
    r_mux: Dict[str, float] = {}
    mux_csv: Optional[Path] = None
    rr_shared = None
    if not bool(int(args.build_only)):
        if bool(int(args.rr_reuse_across_cases)):
            _write_live_status(
                progress_status,
                {
                    "stage": "simulate_prepare",
                    "label": "multiplex",
                    "message": "loading SBML into roadrunner",
                    "rr_reuse_across_cases": True,
                },
            )
            rr_shared = etx._load_rr_with_compile_heartbeat(
                sbml_text=sbml_ext,
                status_json=(progress_status if float(args.progress_chunk_h) > 0.0 else None),
                progress_label="multiplex",
            )
            route1._configure_rr(rr_shared)
            df_mux, m_mux, r_mux = etx._run_case_with_rr(
                rr=rr_shared,
                protein_by_gene=protein_by_gene,
                dna_by_gene=dna_by_gene,
                batch_params=batch_params,
                t_end_h=float(args.t_end_h),
                points=int(args.points),
                init_overrides=dna_mux,
                progress_chunk_h=float(args.progress_chunk_h),
                progress_points_per_chunk=int(args.progress_points_per_chunk),
                progress_status_json=(progress_status if float(args.progress_chunk_h) > 0.0 else None),
                progress_live_csv=(progress_live_csv if float(args.progress_chunk_h) > 0.0 else None),
                progress_checkpoint_csv=(progress_ckpt_csv if float(args.progress_chunk_h) > 0.0 else None),
                progress_label="multiplex",
            )
        else:
            sbml_multi = etx._set_dna_initials(sbml_ext, dna_mux)
            df_mux, m_mux, r_mux = etx._run_case(
                sbml_text=sbml_multi,
                protein_by_gene=protein_by_gene,
                dna_by_gene=dna_by_gene,
                batch_params=batch_params,
                t_end_h=float(args.t_end_h),
                points=int(args.points),
                progress_chunk_h=float(args.progress_chunk_h),
                progress_points_per_chunk=int(args.progress_points_per_chunk),
                progress_status_json=(progress_status if float(args.progress_chunk_h) > 0.0 else None),
                progress_live_csv=(progress_live_csv if float(args.progress_chunk_h) > 0.0 else None),
                progress_checkpoint_csv=(progress_ckpt_csv if float(args.progress_chunk_h) > 0.0 else None),
                progress_label="multiplex",
            )
        mux_csv = outdir / "multigene_explicit_tx_hybrid_multiplex.csv"
        df_mux.to_csv(mux_csv, index=False)

    crosstalk_rows: List[Dict[str, object]] = []
    single_metrics: Dict[str, Dict[str, float]] = {}
    if bool(int(args.run_single_controls)) and (not bool(int(args.build_only))):
        for g in gene_cfgs:
            dna_single_name = _single_control_dna(gene_names, dna_mux_by_name, target_gene=g.name, mode=str(args.single_dna_mode))
            dna_single = {"DNA": float(native_dna)}
            for gg in gene_cfgs:
                sid = dna_by_gene[gg.name]
                dna_single[sid] = float(max(0.0, dna_single_name.get(gg.name, 0.0)))

            if rr_shared is not None:
                df_single, m_single, _ = etx._run_case_with_rr(
                    rr=rr_shared,
                    protein_by_gene=protein_by_gene,
                    dna_by_gene=dna_by_gene,
                    batch_params=batch_params,
                    t_end_h=float(args.t_end_h),
                    points=int(args.points),
                    init_overrides=dna_single,
                )
            else:
                sbml_single = etx._set_dna_initials(sbml_ext, dna_single)
                df_single, m_single, _ = etx._run_case(
                    sbml_text=sbml_single,
                    protein_by_gene=protein_by_gene,
                    dna_by_gene=dna_by_gene,
                    batch_params=batch_params,
                    t_end_h=float(args.t_end_h),
                    points=int(args.points),
                )
            single_csv = outdir / f"multigene_explicit_tx_hybrid_single_{route1.wf.sanitize_id(g.name)}.csv"
            df_single.to_csv(single_csv, index=False)
            single_metrics[g.name] = m_single[g.name]

            mux_6 = float(m_mux[g.name]["protein_6h_uM"])
            single_6 = float(m_single[g.name]["protein_6h_uM"])
            ratio = mux_6 / max(single_6, 1e-12)
            crosstalk_rows.append(
                {
                    "gene": g.name,
                    "multiplex_p6_uM": mux_6,
                    "single_p6_uM": single_6,
                    "multiplex_to_single_ratio": float(ratio),
                    "suppression_fraction": float(1.0 - ratio),
                }
            )

    crosstalk_csv: Optional[Path] = None
    if len(crosstalk_rows) > 0:
        df_ct = pd.DataFrame(crosstalk_rows).sort_values("suppression_fraction", ascending=False)
        crosstalk_csv = outdir / "multigene_explicit_tx_hybrid_crosstalk.csv"
        df_ct.to_csv(crosstalk_csv, index=False)

    counts_final = {
        "species_total": int(counts_bg.get("species_total", 0)) + int(delta.get("species_added", 0)),
        "reactions_total": int(counts_bg.get("reactions_total", 0)) + int(delta.get("reactions_added", 0)),
    }

    summary = {
        "workflow": "route1_multigene_explicit_tx_hybrid",
        "assumptions": {
            "explicit_transcription_chain": "generated per gene from CDS with explicit tx intermediates",
            "independent_dna_mrna_lines": True,
            "shared_translation_resources": ["Ribosome", "ATP", "GTP", "AA20"],
            "stiffness_coarse_grain": {
                "tx_compress_scale": float(args.tx_compress_scale),
                "tx_fast_scale": float(args.tx_fast_scale),
                "ribo_bind_scale": float(args.ribo_bind_scale),
            },
            "native_degfp_share": float(args.native_degfp_share),
            "dna_total_M": float(dna_total_M),
            "dna_native_M": float(native_dna),
            "dna_explicit_total_M": float(active_dna),
            "dna_multiplex_alloc_M": {g.name: float(dna_mux_by_name.get(g.name, 0.0)) for g in gene_cfgs},
            "genes": [
                {
                    "name": g.name,
                    "aa_length": int(g.aa_length),
                    "tl_scale": float(g.tl_scale),
                    "tx_scale": float(g.tx_scale),
                    "has_cds_dna": bool(g.cds_dna),
                    "cds_nt_len": int(len(etx._clean_dna(g.cds_dna or ""))) if g.cds_dna else 0,
                    "tx_nt_steps": (None if g.tx_nt_steps is None else int(g.tx_nt_steps)),
                    "dna_M": float(dna_mux_by_name.get(g.name, 0.0)),
                }
                for g in gene_cfgs
            ],
            "transcription_sequence_meta": tx_seq_meta,
            "translation_aa_profile_meta": aa_profile_meta,
            "mrna_decay_base_s-1": float(args.mrna_decay_base),
            "mrna_bound_decay_ratio": float(args.mrna_bound_decay_ratio),
        },
        "counts_background_batch": counts_bg,
        "counts_delta": delta,
        "counts_final": counts_final,
        "counts_explicit_sbml": explicit_counts_now,
        "batch_params_source": str(args.summary_json),
        "batch_params_used": batch_params,
        "hf_meta": hf_meta,
        "execution": {
            "model_cache": {
                "enabled": bool(int(args.use_model_cache)),
                "cache_hit": bool(cache_hit),
                "model_key": str(model_key),
                "cache_dir": str(cache_dir),
                "cache_bundle_path": (None if cache_bundle_path is None else str(cache_bundle_path)),
                "cache_sbml_path": (None if cache_sbml_path is None else str(cache_sbml_path)),
            },
            "rr_reuse_across_cases": bool(int(args.rr_reuse_across_cases)),
            "singleton_lock": bool(int(args.singleton_lock)),
        },
        "metrics": {
            "multiplex_protein_metrics": m_mux,
            "multiplex_resource_metrics": r_mux,
            "single_protein_metrics": single_metrics,
        },
        "files": {
            "multiplex_csv": (None if mux_csv is None else str(mux_csv)),
            "crosstalk_csv": (None if crosstalk_csv is None else str(crosstalk_csv)),
            "progress_status_json": (
                None
                if (bool(int(args.build_only)) or float(args.progress_chunk_h) <= 0.0)
                else str(outdir / "multigene_explicit_tx_hybrid_progress_status.json")
            ),
            "progress_live_csv": (
                None
                if (bool(int(args.build_only)) or float(args.progress_chunk_h) <= 0.0)
                else str(outdir / "multigene_explicit_tx_hybrid_live.csv")
            ),
            "progress_checkpoint_csv": (
                None
                if (bool(int(args.build_only)) or float(args.progress_chunk_h) <= 0.0)
                else str(outdir / "multigene_explicit_tx_hybrid_checkpoints.csv")
            ),
        },
    }
    summary_path = outdir / "multigene_explicit_tx_hybrid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("Saved:")
    print(summary_path)
    if mux_csv is not None:
        print(mux_csv)
    if crosstalk_csv is not None:
        print(crosstalk_csv)
    if bool(int(args.run_single_controls)):
        for g in gene_cfgs:
            print(outdir / f"multigene_explicit_tx_hybrid_single_{route1.wf.sanitize_id(g.name)}.csv")


if __name__ == "__main__":
    main()
