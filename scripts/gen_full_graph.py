"""
    Module for generating the full scene graph from the initial 3DSG .
        - Spatial Relation Decision using LLM
        - Functional Relation Reasoning using LLM
        - Affordance Reasoning using LLM

    Example:
        $ python scripts/gen_full_graph.py scene_id=0kitchen/video0 dataset=FunGraph3D save_folder_name=
"""
import argparse
import gzip
import io
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from utils.general_utils import read_json
from prompts.gpt import GPTprompt
from slam.slam_classes import MapObjectList

# ===== hydra / omegaconf =====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# register custom resolver used in logging filename: ${replace:x,/,-}
OmegaConf.register_new_resolver("replace", lambda s, a, b: str(s).replace(a, b))

LOGGER = logging.getLogger(__name__)  # [HYDRA] use hydra-managed logger

# -----------------------------
# OpenAI Responses API (official SDK)
# -----------------------------
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "OpenAI SDK not found. Install with: pip install openai>=1.51.0"
    ) from e

OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
# Check shapshot in here: https://platform.openai.com/docs/models/gpt-5 
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07")

# =========================
# Config enrichment
# =========================
def _resolve_path(p) -> str:
    """Resolve path against original CWD (Hydra changes run dir)."""
    pp = Path(str(p))
    if not pp.is_absolute():
        pp = Path(get_original_cwd()) / pp
    return str(pp)

def _process_cfg(cfg: DictConfig) -> None:  # [HYDRA]
    """
    - require scene_id & dataset
    - attach dataset_root / dataset_config
    """
    if not cfg.get("scene_id") or not cfg.get("dataset"):
        raise ValueError("Both `scene_id` and `dataset` are required. e.g., scene_id=0kitchen/video0 dataset=FunGraph3D")
    if str(cfg.dataset) not in cfg.ALLOWED_DATASETS:
        raise ValueError(f"`dataset` must be one of {sorted(cfg.ALLOWED_DATASETS)}; got {cfg.dataset}")

    prev_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.mode = "full"   # full, func

    ds = str(cfg.dataset)
    if ds == "FunGraph3D":
        cfg.dataset_root   = _resolve_path(cfg.FUNGRAPH3D_root)
        cfg.dataset_config = _resolve_path(cfg.FUNGRAPH3D_config_path)
        cfg.mode = "func" # functional relation only for FunGraph3D
    elif ds == "SceneFun3Ddev":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "dev")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
        cfg.mode = "func" # functional relation only for SceneFun3D
    elif ds == "SceneFun3Dtest":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "test")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
        cfg.mode = "func" # functional relation only for SceneFun3D
    elif ds == "PADO":
        cfg.dataset_root   = _resolve_path(cfg.PADO_root)
        pado_cfg_key = "PADO_config" if "PADO_config" in cfg else "PADO_config_path"
        cfg.dataset_config = _resolve_path(cfg[pado_cfg_key])
        cfg.mode = "full"
    
    OmegaConf.set_struct(cfg, prev_struct)

def _idx_from(node_id: str, prefix: str) -> Optional[int]:
    """Parse index from 'obj_12' / 'part_3' etc."""
    if not isinstance(node_id, str) or not node_id.startswith(prefix + "_"):
        return None
    try:
        return int(node_id.split("_", 1)[1])
    except Exception:
        return None

def _load_mol(serialized) -> MapObjectList:
    mol = MapObjectList()
    mol.load_serializable(serialized)
    return mol

def _as_edge(a: str, b: str, label: str) -> Optional[tuple]:
    ai = _idx_from(a, "obj")
    aj = _idx_from(a, "part")
    bi = _idx_from(b, "obj")
    bj = _idx_from(b, "part")
    if ai is not None and bj is not None:
        return (ai, bj, -1, label)   # obj–part
    if ai is not None and bi is not None:
        return (ai, -1, bi, label)   # obj–obj
    if aj is not None and bi is not None:
        return (bi, aj, -1, label)   # part–obj (normalize as obj–part)
    return None

@hydra.main(version_base=None, config_path="../configs", config_name="CAPA")
def main(cfg: DictConfig):  
    LOGGER.info("START main()")
    _process_cfg(cfg)

    LOGGER.info("Loading inital 3D Scene Graph json file")
    sg_init_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'scene_graph' / f"initial_3d_scene_graph.json"
    initial_graph = read_json(sg_init_path)

    if initial_graph is None or len(initial_graph) == 0:
        LOGGER.error(f"Initial 3D Scene Graph file not found: {sg_init_path}")
    
    if cfg.mode == "func":
        spatial_relation = initial_graph.pop("spatial_relation", None)
        initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
    else:
        initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
    LOGGER.info("Initial Graph Loaded.")

    LOGGER.info("Loading updated fused 3D objects file")
    """
        updated_results = {
            'objects': objects.to_serializable(),
            'cfg': results['cfg'],
            'class_names': results['class_names'],
            'class_colors': results['class_colors'],
            'inter_id_candidate': rigid_inter_id_candidate
        }    
    """
    obj_pkl_in = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'object' / 'pcd_saves' / f"full_pcd_ram_update.pkl.gz"
    with gzip.open(obj_pkl_in, "rb") as f:
        obj_results = pickle.load(f)

    LOGGER.info("Loading updated fused 3D parts file")
    """
        updated_results = {
            'objects': parts.to_serializable(),
            'cfg': part_results['cfg'],
            'class_names': part_results['class_names'],
            'class_colors': part_results['class_colors'],
            'part_inter_id_candidate': part_inter_id_candidate
        }    
    """
    part_pkl_in = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'part' / 'pcd_saves' / f"full_pcd_ram_update.pkl.gz"
    with gzip.open(part_pkl_in, "rb") as f:
        part_results = pickle.load(f)
    
    if cfg.mode == "func":
        LOGGER.info("Generating Functional 3D Scene Graph without Spatial Relations")
        LOGGER.info("API REQUESTING...")
        resp = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=GPTprompt().system_prompt,
            input= initial_graph,
            reasoning={"effort":"high"},
            text={
                "verbosity":"high",
                "format": {
                    "type": "json_schema",
                    "name": "3D_Scene_Graph",
                    "strict": True,
                    # ------------ SCHEMA ------------ 
                    "schema": {
                        "type": "object",
                        "properties": {
                            # ------------ for Objects ------------ 
                            "objects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "pattern":"^(obj|part)_\\d+$"
                                            },
                                        "label": {"type": "string"},
                                        "connected_parts": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern":"^part_\\d+$"
                                                }
                                        }
                                    },
                                    "required": ["id", "label", "connected_parts"],
                                    "additionalProperties": False
                                }
                            },
                            # ------------ for Parts ------------ 
                            "parts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "string",
                                            "pattern":"^part_\\d+$"
                                            },
                                        "label":   {"type": "string"}
                                    },
                                    "required": ["id", "label"],
                                    "additionalProperties": False
                                }
                            },
                            # ------------ for Functional Relations ------------ 
                            "functional_relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pair": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern":"^(obj|part)_\\d+$",
                                                },
                                            "minItems": 2, 
                                            "maxItems": 2
                                        },
                                        "label": {"type": "string"},
                                        "reason": {
                                            "type": "string",
                                            "description": "Reason for inferring the relations"
                                            },
                                        "score": {
                                            "type": "number",
                                            "description": "Confidence score of the inferred relation",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "multipleOf": 0.1
                                            },
                                    },
                                    "required": ["pair", "label", "reason", "score"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["objects", "parts", "functional_relations"],
                        "additionalProperties": False
                    }
                }   
            }
        )

        # llm_result = resp.output_text.strip().replace("'", "\'")
        try:
            # llm_result = json.loads(llm_result)
            llm_result = json.loads(resp.output_text)
            LOGGER.info("Successfully parsed LLM response as JSON.")
        except Exception as e:
            LOGGER.error("Failed to parse LLM response as JSON.")
            LOGGER.error(f"Response: {resp}")
            raise e
        
        try:
            LOGGER.debug(f"reasoning: {resp.reasoning}")
            LOGGER.debug(f"usage: {resp.usage}")
        except Exception:
            pass

        objs = llm_result["objects"]
        parts = llm_result["parts"]
        func_rels = llm_result["functional_relations"]

        # update full_pcd_ram_update.pkl.gz and save to full_pcd_ram_llm.pkl.gz for both objects and parts
        # generate edge pickle file to evaluation
        obj_mol  = _load_mol(obj_results["objects"])
        part_mol = _load_mol(part_results["objects"])

        # Update Objects
        for item in objs:
            oid = item.get("id")        # obj_00
            label = item.get("label")   
            oi = _idx_from(oid, "obj")  # 00
            if oi is None:
                LOGGER.warning(f"[objects] skip invalid id: {oid}")
                continue
            try:
                obj_entry = obj_mol[oi]
            except Exception:
                LOGGER.warning(f"[objects] index out of range: {oi}")
                continue
            if label:
                if obj_entry["refined_obj_tag"] != label:
                    refined = obj_entry['refined_obj_tag']
                    LOGGER.debug(f"[objects] {oid} : {refined}-> {label}")
                obj_entry["refined_obj_tag"] = label

            cps = item.get("connected_parts", None)
            if isinstance(cps, list):
                cp_idx = []
                for pid in cps:
                    pj = _idx_from(pid, "part")
                    if pj is None:
                        LOGGER.warning(f"[objects] skip invalid part id: {pid}")
                        continue
                    # range check (only if we can index)
                    try:
                        _ = part_mol[pj]
                    except Exception:
                        LOGGER.warning(f"[objects] part index OOR: {pj}")
                        continue
                    cp_idx.append(int(pj))
                obj_entry["connected_parts"] = sorted(set(cp_idx))

        # Update Parts
        for item in parts:
            pid = item.get("id")
            label = item.get("label")
            pj = _idx_from(pid, "part")
            if pj is None:
                LOGGER.warning(f"[parts] skip invalid id: {pid}")
                continue
            try:
                part_entry = part_mol[pj]
            except Exception:
                LOGGER.warning(f"[parts] index out of range: {pj}")
                continue
            if label:
                if part_entry["refined_obj_tag"] != label:
                    refined = part_entry['refined_obj_tag']
                    LOGGER.debug(f"[parts] {pid} : {refined}-> {label}")
                part_entry["refined_obj_tag"] = label


        # Save updated object/part mol with LLM labels
        LOGGER.info("Saving updated object/part to serialized mol")
        obj_results["objects"] = obj_mol.to_serializable()
        part_results["objects"] = part_mol.to_serializable()

        obj_pkl_out  = obj_pkl_in.with_name("full_pcd_ram_llm.pkl.gz")
        part_pkl_out = part_pkl_in.with_name("full_pcd_ram_llm.pkl.gz")
        with gzip.open(obj_pkl_out, "wb") as f:
            pickle.dump(obj_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(part_pkl_out, "wb") as f:
            pickle.dump(part_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(f"[PKL] Saved objects: {obj_pkl_out}")
        LOGGER.info(f"[PKL] Saved parts  : {part_pkl_out}")

        # Generate edge pickle file
        edges = []
        for rel in func_rels:
            pair = rel.get("pair", [])
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            lab = rel.get("label", "")
            e = _as_edge(pair[0], pair[1], lab)
            if e is not None:
                edges.append(e)
        # unique
        edges = list({(e[0], e[1], e[2], e[3]) for e in edges})

        edge_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "cfslam_funcgraph_edges.pkl"
        with open(edge_path, "wb") as f:
            pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(f"[Edges] Saved: {edge_path} (count={len(edges)})")

        # Save functional 3D scene graph json file
        sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
        sg_dir.mkdir(parents=True, exist_ok=True)
        sg_out = {
            "object": {},
            "part": {},
            "functional_relation": [{"pair": [f"{'obj_'+str(e[0])}", (f'part_{e[1]}' if e[1]!=-1 else f"obj_{e[2]}")], "label": e[3]} for e in edges],
        }

        # fill nodes from current PKL state
        # objects
        src_obj = obj_mol if hasattr(obj_mol, "__len__") else []
        for i in range(len(src_obj)):
            ent = src_obj[i]
            cps = ent.get("connected_parts", [])
            sg_out["object"][f"obj_{i}"] = {
                "label": ent.get("refined_obj_tag", ent.get("majority_class_name", "")),
                "center": list(ent["bbox"].center) if "bbox" in ent else None,
                "extent": list(ent["bbox"].extent) if "bbox" in ent else None,
                "connected_parts": [f"part_{j}" for j in cps] if isinstance(cps, list) else []
            }
        # parts
        src_part = part_mol if hasattr(part_mol, "__len__") else []
        for j in range(len(src_part)):
            ent = src_part[j]
            sg_out["part"][f"part_{j}"] = {
                "label": ent.get("refined_obj_tag", ent.get("majority_class_name", "")),
                "center": list(ent["bbox"].center) if "bbox" in ent else None,
                "extent": list(ent["bbox"].extent) if "bbox" in ent else None,
            }

        sg_path = sg_dir / "functional_3d_scene_graph.json"
        with open(sg_path, "w") as jf:
            json.dump(sg_out, jf, ensure_ascii=False, indent=2)
        LOGGER.info(f"[SceneGraph] Saved: {sg_path}")


    elif cfg.mode == "full":
        LOGGER.info("Generating Full 3D Scene Graph with Spatial and Functional Relations")
        # TODO: implement full graph generation
        pass
    else:
        LOGGER.error(f"Unknown mode: {cfg.mode}")

    LOGGER.info("FINISH main()")

if __name__ == "__main__":
    main()