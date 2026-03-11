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
import time
from pathlib import Path
from typing import Dict, Any, Optional
from utils.general_utils import read_json
from prompts.gpt import GPTprompt
from slam.slam_classes import MapObjectList
from time import sleep

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

    cfg.mode = None

    ds = str(cfg.dataset)
    if ds == "FunGraph3D":
        cfg.dataset_root   = _resolve_path(cfg.FUNGRAPH3D_root)
        cfg.dataset_config = _resolve_path(cfg.FUNGRAPH3D_config_path)
        # cfg.mode = "func" # functional relation only for FunGraph3D
        cfg.mode = "uni"
    elif ds == "SceneFun3Ddev":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "dev")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
        # cfg.mode = "func" # functional relation only for SceneFun3D
        cfg.mode = "uni"
    elif ds == "SceneFun3Dtest":
        cfg.dataset_root   = _resolve_path(Path(cfg.SCENEFUN3D_root) / "test")
        cfg.dataset_config = _resolve_path(cfg.SCENEFUN3D_config)
        # cfg.mode = "func" # functional relation only for SceneFun3D
        cfg.mode = "uni"
    elif ds == "CAPAD":
        cfg.dataset_root   = _resolve_path(cfg.CAPAD_root)
        cfg.dataset_config = _resolve_path(cfg.CAPAD_config)
        cfg.mode = "ca"
    elif ds == "ReplicaSSG":
        cfg.dataset_root   = _resolve_path(cfg.ReplicaSSG_root)
        cfg.dataset_config = _resolve_path(cfg.ReplicaSSG_config)
        cfg.mode = "spat"
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    
    
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

def load_node(cfg):
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
    
    return obj_results, part_results

@hydra.main(version_base=None, config_path="../configs", config_name="CAPA")
def main(cfg: DictConfig):  
    LOGGER.info("START main()")
    _process_cfg(cfg)

    LOGGER.info(f"Generate 3DSG with {cfg.mode} mode")

    LOGGER.info("Loading inital 3D Scene Graph json file")
    sg_init_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / 'scene_graph' / f"initial_3d_scene_graph.json"
    initial_graph = read_json(sg_init_path)

    if initial_graph is None or len(initial_graph) == 0:
        LOGGER.error(f"Initial 3D Scene Graph file not found: {sg_init_path}")

    if cfg.mode == "func":
        LOGGER.info("Generating Functional 3D Scene Graph")
        if cfg.dataset == "FunGraph3D" or cfg.dataset.startswith("SceneFun3D"):

            # Preprocess initial graph: remove spatial relations & clean up object/part centers/extents
            spatial_relation = initial_graph.pop("spatial_relation", None)

            if "object" in initial_graph and isinstance(initial_graph["object"], dict):
                for obj_id, obj in initial_graph["object"].items():
                    if "center" in obj:
                        obj["center"] = [round(float(num), 2) for num in obj["center"]]
                    obj.pop("extent", None)

            if "part" in initial_graph and isinstance(initial_graph["part"], dict):
                for part_id, part in initial_graph["part"].items():
                    if "center" in part:
                        part["center"] = [round(float(num), 2) for num in part["center"]]
                    part.pop("extent", None)

            initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
        else:
            initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
        LOGGER.info("Initial Graph Loaded.")

        obj_results, part_results = load_node(cfg)

        prompts = GPTprompt(config=cfg)
        LOGGER.info("Generating Functional 3D Scene Graph without Spatial Relations")
        LOGGER.info("API REQUESTING... (Local Functional Relations)")
        resp = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=prompts.system_prompt_local,
            input=initial_graph,
            background=True,
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

        LOGGER.info("API REQUESTING... (Remote Functional Relations)")
        resp2 = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=prompts.system_prompt_remote,
            input=initial_graph,
            background=True,
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


        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        jobs = {"local": resp.id, "remote": resp2.id}
        results = {}
        prev_status = {}  # name -> last_status
        
        timeout = 1200  # seconds
        start_time = time.time()

        while jobs:
            for name, job_id in list(jobs.items()):
                r = client.responses.retrieve(job_id)
                s = getattr(r, "status", None)

                if prev_status.get(name) != s:
                    LOGGER.info(f"[{name}] status changed: {prev_status.get(name)} -> {s}")
                    prev_status[name] = s

                if s in {"completed", "succeeded"}:
                    results[name] = r
                    del jobs[name]
                elif s in {"failed", "errored", "cancelled", "incomplete"}:
                    LOGGER.error(f"[{name}] job failed with status={s}")
                    raise RuntimeError(f"{name} job failed: {s}") 
                
                if time.time() - start_time > timeout:
                    LOGGER.error(f"[{name}] job timeout after {timeout} seconds")
                    raise TimeoutError(f"{name} job timeout")
            sleep(5)

        resp  = results["local"]
        resp2 = results["remote"]

        
        try:
            # llm_result = json.loads(llm_result)
            llm_result_local = json.loads(resp.output_text)
            LOGGER.info("Successfully parsed LLM response as JSON - Local Functional Relations")
            llm_result_remote = json.loads(resp2.output_text)
            LOGGER.info("Successfully parsed LLM response as JSON - Remote Functional Relations")
        except Exception as e:
            LOGGER.error("Failed to parse LLM response as JSON.")
            LOGGER.error(f"Response: {resp}")
            LOGGER.error(f"Response: {resp2}")
            raise e
        
        try:
            LOGGER.debug(f"reasoning: {resp.reasoning}")
            LOGGER.debug(f"usage: {resp.usage}")
            LOGGER.debug(f"reasoning: {resp2.reasoning}")
            LOGGER.debug(f"usage: {resp2.usage}")
        except Exception:
            pass


        objs  = llm_result_local.get("objects")
        parts = llm_result_local.get("parts")
        func_rels_local = llm_result_local["functional_relations"]
        func_rels_remote = llm_result_remote["functional_relations"]

        owner = {}
        for obj in objs:
            cps = obj.get("connected_parts", [])
            if isinstance(cps, list):
                for pid in cps:
                    if isinstance(pid, str) and pid.startswith("part_"):
                        owner[pid] = obj.get("id")

        func_rels = []
        _seen_pairs = set()   # (tuple(pair), label)

        def _push_relation(rel):
            pair = rel.get("pair", [])
            lab  = rel.get("label", "")
            if not (isinstance(pair, list) and len(pair) == 2):
                return
            key = (tuple(pair), lab)
            if key not in _seen_pairs:
                _seen_pairs.add(key)
                func_rels.append(rel)

        # Add local functional relations
        for r in func_rels_local:
            _push_relation(r)

        # Add remote functional relations, and propagate part-object to object-object
        for r in func_rels_remote:
            _push_relation(r)
            pair = r.get("pair", [])
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            a, b = pair[0], pair[1]
            if isinstance(a, str) and a.startswith("part_") and isinstance(b, str) and b.startswith("obj_"):
                parent = owner.get(a)
                if parent and parent != b:
                    rr = dict(r)
                    rr["pair"]   = [parent, b]
                    rr["reason"] = (r.get("reason","") + " (propagated from part to object)").strip()
                    rr["score"]  = float(max(0.0, min(1.0, r.get("score", 0.0) - 0.05)))
                    _push_relation(rr)
            elif isinstance(a, str) and a.startswith("obj_") and isinstance(b, str) and b.startswith("part_"):
                parent = owner.get(b)
                if parent and parent != a:
                    rr = dict(r)
                    rr["pair"]   = [a, parent]
                    rr["reason"] = (r.get("reason","") + " (propagated from object to part)").strip()
                    rr["score"]  = float(max(0.0, min(1.0, r.get("score", 0.0) - 0.05)))
                    _push_relation(rr)

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
        edges_desc = []
        for rel in func_rels:
            pair = rel.get("pair", [])
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            lab = rel.get("label", "")
            e = _as_edge(pair[0], pair[1], lab)
            if e is not None:
                edges.append(e)
                reason = rel.get("reason", "")
                score = rel.get("score", -1.0)
                edges_desc.append((e, reason, score))
        
        edge_meta = {}
        for e, reason, score in edges_desc:
            # key: the edge tuple; val: (reason, score)
            edge_meta[(e[0], e[1], e[2], e[3])] = (reason, float(score))
        
        # unique + deterministic order
        edges = sorted(edge_meta.keys())

        edge_path = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "cfslam_funcgraph_edges.pkl"
        with open(edge_path, "wb") as f:
            pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(f"[Edges] Saved: {edge_path} (count={len(edges)})")

        # Save functional 3D scene graph json file
        sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
        sg_dir.mkdir(parents=True, exist_ok=True)

        func_rel = []
        for (oi, pj, ok, lab) in edges:
            a = f"obj_{oi}"
            b = f"part_{pj}" if pj != -1 else f"obj_{ok}"
            reason, score = edge_meta.get((oi, pj, ok, lab), ("", -1.0))
            func_rel.append({
                "pair":  [a, b],
                "label": lab,
                "reason": reason,
                "score": float(score),
            })
        sg_out = {
            "object": {},     
            "part":   {},
            "functional_relation": func_rel
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


    elif cfg.mode == "ca":
        LOGGER.info("Generating Context-aware 3D Scene Graph with Spatial, Functional Relations, and Affordances")

        # -----------------------------
        # Load scenario instructions
        # -----------------------------
        scenario_path = (Path(cfg.dataset_root) / cfg.scene_id).parent / "scenario.json"
        if not scenario_path.exists():
            LOGGER.error(f"Scenario file not found: {scenario_path}")
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        scenario = read_json(scenario_path)
        LOGGER.info(f"Scenario loaded from: {scenario_path}")

        instructions = []
        for i in scenario["scenario"].keys():
            sdata = scenario["scenario"][i]
            for j in sdata["route"].keys():
                instructions.append([sdata["scenario_goal"], sdata["route"][j]["instruction"]])

        prompts = GPTprompt(config=cfg)

        sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
        sg_dir.mkdir(parents=True, exist_ok=True)

        merged_all = []

        for step_idx, (goal, inst) in enumerate(instructions):
            # Combine goal + step instruction as a single instruction string
            instruction = str(inst)
            if goal:
                instruction = (str(goal).rstrip(".") + ". " + instruction).strip()

            # graph_in = dict(initial_graph)
            # graph_in["instruction"] = instruction
            # graph_str = json.dumps(graph_in, ensure_ascii=False, separators=(",", ":"))
                        # -----------------------------
            # Prepare per-agent input graphs (CA mode)
            # -----------------------------
            nd = 3  # decimal precision (same as mode=func)

            # (A) local / remote input: no spatial_relation, no extent, rounded centers (+ instruction)
            graph_func = {"object": {}, "part": {}, "instruction": instruction}

            if isinstance(initial_graph, dict) and isinstance(initial_graph.get("object"), dict):
                for oid, obj in initial_graph["object"].items():
                    if not isinstance(obj, dict):
                        continue
                    o = {
                        "label": obj.get("label", ""),
                        "connected_parts": obj.get("connected_parts", []) if isinstance(obj.get("connected_parts"), list) else [],
                    }
                    if isinstance(obj.get("center"), list):
                        o["center"] = [round(float(x), nd) for x in obj["center"]]
                    graph_func["object"][oid] = o

            if isinstance(initial_graph, dict) and isinstance(initial_graph.get("part"), dict):
                for pid, part in initial_graph["part"].items():
                    if not isinstance(part, dict):
                        continue
                    p = {"label": part.get("label", "")}
                    if isinstance(part.get("center"), list):
                        p["center"] = [round(float(x), nd) for x in part["center"]]
                    graph_func["part"][pid] = p

            graph_str_func = json.dumps(graph_func, ensure_ascii=False, separators=(",", ":"))

            # (B) spatial input: keep center+extent, include spatial_relation prior candidates, no functional relations
            graph_spatial = {"object": {}, "part": {}, "instruction": instruction}

            if isinstance(initial_graph, dict) and ("spatial_relation" in initial_graph):
                graph_spatial["spatial_relation"] = initial_graph.get("spatial_relation")

            if isinstance(initial_graph, dict) and isinstance(initial_graph.get("object"), dict):
                for oid, obj in initial_graph["object"].items():
                    if not isinstance(obj, dict):
                        continue
                    o = {
                        "label": obj.get("label", ""),
                        "connected_parts": obj.get("connected_parts", []) if isinstance(obj.get("connected_parts"), list) else [],
                    }
                    if isinstance(obj.get("center"), list):
                        o["center"] = [round(float(x), nd) for x in obj["center"]]
                    if isinstance(obj.get("extent"), list):
                        o["extent"] = [round(float(x), nd) for x in obj["extent"]]
                    graph_spatial["object"][oid] = o

            if isinstance(initial_graph, dict) and isinstance(initial_graph.get("part"), dict):
                for pid, part in initial_graph["part"].items():
                    if not isinstance(part, dict):
                        continue
                    p = {"label": part.get("label", "")}
                    if isinstance(part.get("center"), list):
                        p["center"] = [round(float(x), nd) for x in part["center"]]
                    if isinstance(part.get("extent"), list):
                        p["extent"] = [round(float(x), nd) for x in part["extent"]]
                    graph_spatial["part"][pid] = p

            graph_str_spatial = json.dumps(graph_spatial, ensure_ascii=False, separators=(",", ":"))

            # (C) affordance input: no relation info (reuse func-style input)
            graph_str_aff = graph_str_func


            LOGGER.info(f"[CA] API REQUESTING... step={step_idx} (Local / Remote / Spatial / Affordance)")

            # -----------------------------
            # (1) Local functional relations
            # -----------------------------
            LOGGER.info("API REQUESTING... (Local Functional Relations)")
            resp = client.responses.create(
                model=OPENAI_CHAT_MODEL,
                instructions=prompts.system_prompt_local,
                input=graph_str_func,
                background=True,
                reasoning={"effort": "high"},
                text={
                    "verbosity": "high",
                    "format": {
                        "type": "json_schema",
                        "name": "CA3DSG_local",
                        "strict": True,
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
                                                    "pattern": "^part_\\d+$"
                                                },
                                            },
                                        },
                                        "required": ["id", "label", "connected_parts"],
                                        "additionalProperties": False,
                                    },
                                },
                                # ------------ for Parts ------------
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {
                                                "type": "string", 
                                                "pattern": "^part_\\d+$"
                                                },
                                            "label": {"type": "string"},
                                        },
                                        "required": ["id", "label"],
                                        "additionalProperties": False,
                                    },
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
                                                    "pattern": "^(obj|part)_\\d+$"
                                                    },
                                                "minItems": 2,
                                                "maxItems": 2,
                                            },
                                            "label": {"type": "string"},
                                            "reason": {"type": "string"},
                                            "score": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                                "multipleOf": 0.1,
                                            },
                                        },
                                        "required": ["pair", "label", "reason", "score"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["objects", "parts", "functional_relations"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            # -----------------------------
            # (2) Remote functional relations
            # -----------------------------
            LOGGER.info("API REQUESTING... (Remote Functional Relations)")
            resp2 = client.responses.create(
                model=OPENAI_CHAT_MODEL,
                instructions=prompts.system_prompt_remote,
                input=graph_str_func,
                background=True,
                reasoning={"effort": "high"},
                text={
                    "verbosity": "high",
                    "format": {
                        "type": "json_schema",
                        "name": "CA3DSG_remote",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string", "pattern": "^obj_\\d+$"},
                                            "label": {"type": "string"},
                                            "connected_parts": {
                                                "type": "array",
                                                "items": {"type": "string", "pattern": "^part_\\d+$"},
                                            },
                                        },
                                        "required": ["id", "label", "connected_parts"],
                                        "additionalProperties": False,
                                    },
                                },
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string", "pattern": "^part_\\d+$"},
                                            "label": {"type": "string"},
                                        },
                                        "required": ["id", "label"],
                                        "additionalProperties": False,
                                    },
                                },
                                "remote_relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "pair": {
                                                "type": "array",
                                                "items": {"type": "string", "pattern": "^(obj|part)_\\d+$"},
                                                "minItems": 2,
                                                "maxItems": 2,
                                            },
                                            "label": {"type": "string"},
                                            "reason": {"type": "string"},
                                            "score": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                                "multipleOf": 0.1,
                                            },
                                        },
                                        "required": ["pair", "label", "reason", "score"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["objects", "parts", "remote_relations"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            # -----------------------------
            # (3) Spatial relations (object–object only)
            # -----------------------------
            LOGGER.info("API REQUESTING... (Spatial Relations)")
            resp3 = client.responses.create(
                model=OPENAI_CHAT_MODEL,
                instructions=prompts.system_prompt_spatial,
                input=graph_str_spatial,
                background=True,
                reasoning={"effort": "high"},
                text={
                    "verbosity": "high",
                    "format": {
                        "type": "json_schema",
                        "name": "CA3DSG_spatial",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string", "pattern": "^obj_\\d+$"},
                                            "label": {"type": "string"},
                                            "connected_parts": {
                                                "type": "array",
                                                "items": {"type": "string", "pattern": "^part_\\d+$"},
                                            },
                                        },
                                        "required": ["id", "label", "connected_parts"],
                                        "additionalProperties": False,
                                    },
                                },
                                "parts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string", "pattern": "^part_\\d+$"},
                                            "label": {"type": "string"},
                                        },
                                        "required": ["id", "label"],
                                        "additionalProperties": False,
                                    },
                                },
                                "spatial_relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "pair": {
                                                "type": "array",
                                                "items": {"type": "string", "pattern": "^obj_\\d+$"},
                                                "minItems": 2,
                                                "maxItems": 2,
                                            },
                                            "label": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "minItems": 1,
                                                "maxItems": 5,
                                            },
                                            "scores": {
                                                "type": "array",
                                                "items": {
                                                    "type": "number",
                                                    "minimum": 0.0,
                                                    "maximum": 1.0,
                                                    "multipleOf": 0.1,
                                                },
                                                "minItems": 1,
                                                "maxItems": 5,
                                            },
                                        },
                                        "required": ["pair", "label", "scores"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["objects", "parts", "spatial_relations"],
                            "additionalProperties": False,
                        },
                    },
                },
            )

            # -----------------------------
            # (4) Affordances (open-vocabulary)
            # -----------------------------
            LOGGER.info("API REQUESTING... (Affordance)")
            resp4 = client.responses.create(
                model=OPENAI_CHAT_MODEL,
                instructions=prompts.system_prompt_affordance,
                input=graph_str_aff,
                background=True,
                reasoning={"effort": "high"},
                text={
                    "verbosity": "high",
                    "format": {
                        "type": "json_schema",
                        "name": "CA3DSG_affordance",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "affordance": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string", "pattern": "^(obj|part)_\\d+$"},
                                            "verb": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "minItems": 0
                                            },
                                        },
                                        "required": ["id", "verb"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["affordance"],
                            "additionalProperties": False,
                        },
                    },
                },
            )


            # -----------------------------
            # Wait all 4 jobs
            # -----------------------------
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

            jobs = {"local": resp.id, "remote": resp2.id, "spatial": resp3.id, "aff": resp4.id}
            results = {}
            prev_status = {}

            timeout = 1200  # seconds
            start_time = time.time()

            while jobs:
                for name, job_id in list(jobs.items()):
                    r = client.responses.retrieve(job_id)
                    s = getattr(r, "status", None)

                    if prev_status.get(name) != s:
                        LOGGER.info(f"[{name}] status changed: {prev_status.get(name)} -> {s}")
                        prev_status[name] = s

                    if s in {"completed", "succeeded"}:
                        results[name] = r
                        del jobs[name]
                    elif s in {"failed", "errored", "cancelled", "incomplete"}:
                        LOGGER.error(f"[{name}] job failed with status={s}")
                        raise RuntimeError(f"{name} job failed: {s}")

                    if time.time() - start_time > timeout:
                        LOGGER.error(f"[{name}] job timeout after {timeout} seconds")
                        raise TimeoutError(f"{name} job timeout")

                sleep(5)

            resp = results["local"]
            resp2 = results["remote"]
            resp3 = results["spatial"]
            resp4 = results["aff"]

            try:
                llm_local = json.loads(resp.output_text)
                llm_remote = json.loads(resp2.output_text)
                llm_spatial = json.loads(resp3.output_text)
                llm_aff = json.loads(resp4.output_text)
            except Exception as e:
                LOGGER.error("Failed to parse LLM response as JSON.")
                LOGGER.error(f"Response(local): {resp}")
                LOGGER.error(f"Response(remote): {resp2}")
                LOGGER.error(f"Response(spatial): {resp3}")
                LOGGER.error(f"Response(aff): {resp4}")
                raise e

            # -----------------------------
            # Merge 4 agent outputs
            # - keep nodes that appear in >=2 results
            # - vote label if mismatch
            # -----------------------------
            def _nodes_from_graph(g):
                nodes = {}
                for o in g.get("objects", []):
                    oid = o.get("id")
                    lab = o.get("label")
                    if isinstance(oid, str) and isinstance(lab, str):
                        nodes[oid] = lab
                for p in g.get("parts", []):
                    pid = p.get("id")
                    lab = p.get("label")
                    if isinstance(pid, str) and isinstance(lab, str):
                        nodes[pid] = lab
                return nodes

            nodes_local = _nodes_from_graph(llm_local)
            nodes_remote = _nodes_from_graph(llm_remote)
            nodes_spatial = _nodes_from_graph(llm_spatial)

            nodes_aff = {}
            for it in llm_aff.get("affordance", []):
                if isinstance(it, list) and len(it) >= 2:
                    nid, nlab = it[0], it[1]
                    if isinstance(nid, str) and isinstance(nlab, str):
                        nodes_aff[nid] = nlab

            node_sources = [nodes_local, nodes_remote, nodes_spatial, nodes_aff]

            presence = {}
            for src in node_sources:
                for nid in src.keys():
                    presence[nid] = presence.get(nid, 0) + 1

            keep_ids = set([nid for nid, c in presence.items() if c >= 2])

            def _vote_label(nid: str) -> str:
                votes = {}
                for src in node_sources:
                    lab = src.get(nid)
                    if isinstance(lab, str) and lab:
                        votes[lab] = votes.get(lab, 0) + 1
                if not votes:
                    return ""
                best = max(votes.values())
                cands = [lab for lab, c in votes.items() if c == best]
                if len(cands) == 1:
                    return cands[0]
                llab = nodes_local.get(nid)
                if llab in cands:
                    return llab
                return sorted(cands)[0]

            # prefer local for connected_parts, else initial graph
            local_obj = {o.get("id"): o for o in llm_local.get("objects", []) if isinstance(o, dict)}
            init_obj = initial_graph.get("object", {}) if isinstance(initial_graph, dict) else {}

            obj_ids = [nid for nid in keep_ids if isinstance(nid, str) and nid.startswith("obj_")]
            part_ids = [nid for nid in keep_ids if isinstance(nid, str) and nid.startswith("part_")]

            obj_ids = sorted(
                obj_ids,
                key=lambda x: _idx_from(x, "obj") if _idx_from(x, "obj") is not None else 10**9,
            )
            part_ids = sorted(
                part_ids,
                key=lambda x: _idx_from(x, "part") if _idx_from(x, "part") is not None else 10**9,
            )

            merged_objects = []
            for oid in obj_ids:
                cps = []
                if oid in local_obj and isinstance(local_obj[oid].get("connected_parts"), list):
                    cps = local_obj[oid]["connected_parts"]
                elif isinstance(init_obj, dict) and oid in init_obj and isinstance(init_obj[oid].get("connected_parts"), list):
                    cps = init_obj[oid]["connected_parts"]

                cps = sorted(
                    set([pid for pid in cps if isinstance(pid, str) and pid.startswith("part_") and pid in keep_ids]),
                    key=lambda x: _idx_from(x, "part") if _idx_from(x, "part") is not None else 10**9,
                )
                merged_objects.append({"id": oid, "label": _vote_label(oid), "connected_parts": cps})

            merged_parts = [{"id": pid, "label": _vote_label(pid)} for pid in part_ids]

            # filter relations by kept nodes
            merged_func = []
            _seen = set()
            for r in llm_local.get("functional_relations", []):
                pair = r.get("pair", [])
                lab = r.get("label", "")
                if isinstance(pair, list) and len(pair) == 2 and pair[0] in keep_ids and pair[1] in keep_ids:
                    k = (tuple(pair), lab)
                    if k not in _seen:
                        _seen.add(k)
                        merged_func.append(r)

            merged_remote = []
            _seen = set()
            for r in llm_remote.get("remote_relations", []):
                pair = r.get("pair", [])
                lab = r.get("label", "")
                if isinstance(pair, list) and len(pair) == 2 and pair[0] in keep_ids and pair[1] in keep_ids:
                    k = (tuple(pair), lab)
                    if k not in _seen:
                        _seen.add(k)
                        merged_remote.append(r)

            merged_spatial = []
            _seen = set()
            for r in llm_spatial.get("spatial_relations", []):
                pair = r.get("pair", [])
                lab = r.get("label", [])
                if isinstance(pair, list) and len(pair) == 2 and pair[0] in keep_ids and pair[1] in keep_ids:
                    k = (tuple(pair), tuple(lab) if isinstance(lab, list) else str(lab))
                    if k not in _seen:
                        _seen.add(k)
                        merged_spatial.append(r)

            # affordance: keep only kept nodes, and inherit part affordances to objects
            aff_map = {}
            for it in llm_aff.get("affordance", []):
                if not (isinstance(it, list) and len(it) == 3):
                    continue
                nid, _nlab, verbs = it[0], it[1], it[2]
                if nid not in keep_ids:
                    continue
                if isinstance(verbs, list):
                    for v in verbs:
                        if isinstance(v, str) and v:
                            aff_map.setdefault(nid, set()).add(v)

            owner = {}
            for o in merged_objects:
                for pid in o.get("connected_parts", []):
                    owner[pid] = o.get("id")

            for pid, oid in owner.items():
                if pid in aff_map:
                    aff_map.setdefault(oid, set()).update(aff_map[pid])

            merged_aff = []
            for nid in obj_ids + part_ids:
                verbs = aff_map.get(nid, set())
                if verbs:
                    merged_aff.append([nid, _vote_label(nid), sorted(verbs)])

            merged_graph = {
                "instruction": instruction,
                "objects": merged_objects,
                "parts": merged_parts,
                "functional_relations": merged_func,
                "remote_relations": merged_remote,
                "spatial_relations": merged_spatial,
                "affordance": merged_aff,
            }

            out_path = sg_dir / f"context_aware_3d_scene_graph_step{step_idx:03d}.json"
            with open(out_path, "w") as jf:
                json.dump(merged_graph, jf, ensure_ascii=False, indent=2)
            LOGGER.info(f"[SceneGraph] Saved: {out_path}")

            merged_all.append(merged_graph)

        out_path = sg_dir / "context_aware_3d_scene_graph_all.json"
        with open(out_path, "w") as jf:
            json.dump(merged_all, jf, ensure_ascii=False, indent=2)
        LOGGER.info(f"[SceneGraph] Saved: {out_path}")

    
    elif cfg.mode == "uni":
        LOGGER.info("Generating Unified 3D Scene Graph")
        if cfg.dataset == "FunGraph3D" or cfg.dataset.startswith("SceneFun3D"):

            # Preprocess initial graph: remove spatial relations & clean up object/part centers/extents
            spatial_relation = initial_graph.pop("spatial_relation", None)

            if "object" in initial_graph and isinstance(initial_graph["object"], dict):
                for obj_id, obj in initial_graph["object"].items():
                    if "center" in obj:
                        obj["center"] = [round(float(num), 2) for num in obj["center"]]
                    obj.pop("extent", None)

            if "part" in initial_graph and isinstance(initial_graph["part"], dict):
                for part_id, part in initial_graph["part"].items():
                    if "center" in part:
                        part["center"] = [round(float(num), 2) for num in part["center"]]
                    part.pop("extent", None)

            initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
        else:
            initial_graph = json.dumps(initial_graph, ensure_ascii=False, separators=(",", ":"))
        LOGGER.info("Initial Graph Loaded.")

        obj_results, part_results = load_node(cfg)

        prompts = GPTprompt(config=cfg)
        LOGGER.info("Generating Unified 3D Scene Graph")
        LOGGER.info("API REQUESTING... (Local Functional Relations)")
        resp = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=prompts.system_prompt_local,
            input=initial_graph,
            background=True,
            reasoning={"effort":"high"},
            max_output_tokens=128000,
            text={
                "verbosity": "high",
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
                                            "pattern": "^(obj|part)_\\d+$"
                                        },
                                        "label": {"type": "string"},
                                        "affordance": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 1
                                        },
                                        "connected_parts": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern": "^part_\\d+$"
                                            }
                                        }
                                    },
                                    "required": ["id", "label", "affordance", "connected_parts"],
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
                                            "pattern": "^part_\\d+$"
                                        },
                                        "label": {"type": "string"},
                                        "affordance": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 1
                                        }
                                    },
                                    "required": ["id", "label", "affordance"],
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
                                                "pattern": "^(obj|part)_\\d+$",
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

        LOGGER.info("API REQUESTING... (Remote Functional Relations)")
        resp2 = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=prompts.system_prompt_remote,
            input=initial_graph,
            background=True,
            reasoning={"effort": "high"},
            max_output_tokens=128000,
            text={
                "verbosity": "high",
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
                                            "pattern": "^(obj|part)_\\d+$"
                                        },
                                        "label": {"type": "string"},
                                        "affordance": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 1
                                        },
                                        "connected_parts": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern": "^part_\\d+$"
                                            }
                                        }
                                    },
                                    "required": ["id", "label", "affordance", "connected_parts"],
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
                                            "pattern": "^part_\\d+$"
                                        },
                                        "label": {"type": "string"},
                                        "affordance": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 1
                                        }
                                    },
                                    "required": ["id", "label", "affordance"],
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
                                                "pattern": "^(obj|part)_\\d+$",
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

        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        jobs = {"local": resp.id, "remote": resp2.id}
        results = {}
        prev_status = {}  # name -> last_status

        timeout = 6000  # seconds
        start_time = time.time()

        while jobs:
            for name, job_id in list(jobs.items()):
                r = client.responses.retrieve(job_id)
                s = getattr(r, "status", None)

                if prev_status.get(name) != s:
                    LOGGER.info(f"[{name}] status changed: {prev_status.get(name)} -> {s}")
                    prev_status[name] = s

                if s in {"completed", "succeeded"}:
                    results[name] = r
                    del jobs[name]
                elif s in {"failed", "errored", "cancelled", "incomplete"}:
                    LOGGER.error(f"[{name}] job failed with status={s}")
                    raise RuntimeError(f"{name} job failed: {s}")

                if time.time() - start_time > timeout:
                    LOGGER.error(f"[{name}] job timeout after {timeout} seconds")
                    raise TimeoutError(f"{name} job timeout")
            sleep(5)

        resp = results["local"]
        resp2 = results["remote"]

        try:
            llm_result_local = json.loads(resp.output_text)
            LOGGER.info("Successfully parsed LLM response as JSON - Local Functional Relations")
            llm_result_remote = json.loads(resp2.output_text)
            LOGGER.info("Successfully parsed LLM response as JSON - Remote Functional Relations")
        except Exception as e:
            LOGGER.error("Failed to parse LLM response as JSON.")
            LOGGER.error(f"Response: {resp}")
            LOGGER.error(f"Response: {resp2}")
            raise e

        try:
            LOGGER.debug(f"reasoning: {resp.reasoning}")
            LOGGER.debug(f"usage: {resp.usage}")
            LOGGER.debug(f"reasoning: {resp2.reasoning}")
            LOGGER.debug(f"usage: {resp2.usage}")
        except Exception:
            pass

        objs = llm_result_local.get("objects")
        parts = llm_result_local.get("parts")
        func_rels_local = llm_result_local["functional_relations"]
        func_rels_remote = llm_result_remote["functional_relations"]

        owner = {}
        for obj in objs:
            cps = obj.get("connected_parts", [])
            if isinstance(cps, list):
                for pid in cps:
                    if isinstance(pid, str) and pid.startswith("part_"):
                        owner[pid] = obj.get("id")

        func_rels = []
        _seen_pairs = set()  # (tuple(pair), label)

        def _push_relation(rel):
            pair = rel.get("pair", [])
            lab = rel.get("label", "")
            if not (isinstance(pair, list) and len(pair) == 2):
                return
            key = (tuple(pair), lab)
            if key not in _seen_pairs:
                _seen_pairs.add(key)
                func_rels.append(rel)

        # Add local functional relations
        for r in func_rels_local:
            _push_relation(r)

        # Add remote functional relations, and propagate part-object to object-object
        for r in func_rels_remote:
            _push_relation(r)
            pair = r.get("pair", [])
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            a, b = pair[0], pair[1]
            if isinstance(a, str) and a.startswith("part_") and isinstance(b, str) and b.startswith("obj_"):
                parent = owner.get(a)
                if parent and parent != b:
                    rr = dict(r)
                    rr["pair"] = [parent, b]
                    rr["reason"] = (r.get("reason", "") + " (propagated from part to object)").strip()
                    rr["score"] = float(max(0.0, min(1.0, r.get("score", 0.0) - 0.05)))
                    _push_relation(rr)
            elif isinstance(a, str) and a.startswith("obj_") and isinstance(b, str) and b.startswith("part_"):
                parent = owner.get(b)
                if parent and parent != a:
                    rr = dict(r)
                    rr["pair"] = [a, parent]
                    rr["reason"] = (r.get("reason", "") + " (propagated from object to part)").strip()
                    rr["score"] = float(max(0.0, min(1.0, r.get("score", 0.0) - 0.05)))
                    _push_relation(rr)

        # ---- (3) Build spatial relations from LLM connected_parts (label is always "part of") ----
        spatial_rels = []
        _seen_spatial = set()  # (obj_id, part_id)
        for obj in objs:
            oid = obj.get("id")
            cps = obj.get("connected_parts", [])
            if not (isinstance(oid, str) and oid.startswith("obj_")):
                continue
            if not isinstance(cps, list):
                continue
            for pid in cps:
                if not (isinstance(pid, str) and pid.startswith("part_")):
                    continue
                key = (oid, pid)
                if key not in _seen_spatial:
                    _seen_spatial.add(key)
                    spatial_rels.append({
                        "pair": [oid, pid],
                        "label": "part of"
                    })

        # ---- Keep using fusion nodes for center/extent, but DO NOT save updated PKL ----
        obj_mol = _load_mol(obj_results["objects"])
        part_mol = _load_mol(part_results["objects"])

        # Update Objects (in-memory only: label / connected_parts / affordance)
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
                if obj_entry.get("refined_obj_tag", None) != label:
                    refined = obj_entry.get("refined_obj_tag", "")
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
                    try:
                        _ = part_mol[pj]
                    except Exception:
                        LOGGER.warning(f"[objects] part index OOR: {pj}")
                        continue
                    cp_idx.append(int(pj))
                obj_entry["connected_parts"] = sorted(set(cp_idx))

            aff = item.get("affordance", None)
            aff_list = ["none"]
            if isinstance(aff, list):
                aff_list = [a.strip() for a in aff if isinstance(a, str) and a.strip()]
                if not aff_list:
                    aff_list = ["none"]
            elif isinstance(aff, str) and aff.strip():
                aff_list = [aff.strip()]
            obj_entry["affordance"] = aff_list

        # Update Parts (in-memory only: label / affordance)
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
                if part_entry.get("refined_obj_tag", None) != label:
                    refined = part_entry.get("refined_obj_tag", "")
                    LOGGER.debug(f"[parts] {pid} : {refined}-> {label}")
                part_entry["refined_obj_tag"] = label

            aff = item.get("affordance", None)
            aff_list = ["none"]
            if isinstance(aff, list):
                aff_list = [a.strip() for a in aff if isinstance(a, str) and a.strip()]
                if not aff_list:
                    aff_list = ["none"]
            elif isinstance(aff, str) and aff.strip():
                aff_list = [aff.strip()]
            part_entry["affordance"] = aff_list

        # ---- (2) Remove edge pkl saving; integrate everything into unified_3d_scene_graph.json ----
        sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
        sg_dir.mkdir(parents=True, exist_ok=True)

        # deterministic order for spatial relations (optional, but keeps stable output)
        spatial_rels = sorted(
            spatial_rels,
            key=lambda r: (
                _idx_from(r.get("pair", ["", ""])[0], "obj") if isinstance(r.get("pair", ["", ""])[0], str) and r.get("pair", ["", ""])[0].startswith("obj_") else 10**9,
                _idx_from(r.get("pair", ["", ""])[1], "part") if isinstance(r.get("pair", ["", ""])[1], str) and r.get("pair", ["", ""])[1].startswith("part_") else 10**9,
            )
        )

        # functional relations output (use combined func_rels directly)
        func_rel = []
        func_rels = sorted(
            func_rels,
            key=lambda r: (
                # sort by pair[0]
                (0, _idx_from(r.get("pair", ["", ""])[0], "obj") if _idx_from(r.get("pair", ["", ""])[0], "obj") is not None else 10**9)
                if isinstance(r.get("pair", ["", ""])[0], str) and r.get("pair", ["", ""])[0].startswith("obj_")
                else (1, _idx_from(r.get("pair", ["", ""])[0], "part") if _idx_from(r.get("pair", ["", ""])[0], "part") is not None else 10**9)
                if isinstance(r.get("pair", ["", ""])[0], str) and r.get("pair", ["", ""])[0].startswith("part_")
                else (2, 10**9),
                # sort by pair[1]
                (0, _idx_from(r.get("pair", ["", ""])[1], "obj") if _idx_from(r.get("pair", ["", ""])[1], "obj") is not None else 10**9)
                if isinstance(r.get("pair", ["", ""])[1], str) and r.get("pair", ["", ""])[1].startswith("obj_")
                else (1, _idx_from(r.get("pair", ["", ""])[1], "part") if _idx_from(r.get("pair", ["", ""])[1], "part") is not None else 10**9)
                if isinstance(r.get("pair", ["", ""])[1], str) and r.get("pair", ["", ""])[1].startswith("part_")
                else (2, 10**9),
                r.get("label", "")
            )
        )
        for rel in func_rels:
            pair = rel.get("pair", [])
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            func_rel.append({
                "pair":  [pair[0], pair[1]],
                "label": rel.get("label", ""),
                "reason": rel.get("reason", ""),
                "score": float(rel.get("score", -1.0)),
            })

        sg_out = {
            "object": {},
            "part": {},
            "spatial_relation": spatial_rels,
            "functional_relation": func_rel
        }

        # fill nodes from current fusion state (center/extent) + LLM-updated label/connected_parts/affordance
        # objects
        src_obj = obj_mol if hasattr(obj_mol, "__len__") else []
        for i in range(len(src_obj)):
            ent = src_obj[i]
            cps = ent.get("connected_parts", [])
            aff = ent.get("affordance", ["none"])
            if isinstance(aff, str):
                aff = [aff]
            if not isinstance(aff, list) or not aff:
                aff = ["none"]

            sg_out["object"][f"obj_{i}"] = {
                "label": ent.get("refined_obj_tag", ent.get("majority_class_name", "")),
                "center": list(ent["bbox"].center) if "bbox" in ent else None,
                "extent": list(ent["bbox"].extent) if "bbox" in ent else None,
                "connected_parts": [f"part_{j}" for j in cps] if isinstance(cps, list) else [],
                "affordance": aff
            }

        # parts
        src_part = part_mol if hasattr(part_mol, "__len__") else []
        for j in range(len(src_part)):
            ent = src_part[j]
            aff = ent.get("affordance", ["none"])
            if isinstance(aff, str):
                aff = [aff]
            if not isinstance(aff, list) or not aff:
                aff = ["none"]

            sg_out["part"][f"part_{j}"] = {
                "label": ent.get("refined_obj_tag", ent.get("majority_class_name", "")),
                "center": list(ent["bbox"].center) if "bbox" in ent else None,
                "extent": list(ent["bbox"].extent) if "bbox" in ent else None,
                "affordance": aff
            }

        sg_path = sg_dir / "unified_3d_scene_graph.json"
        with open(sg_path, "w") as jf:
            json.dump(sg_out, jf, ensure_ascii=False, indent=2)
        LOGGER.info(f"[SceneGraph] Saved: {sg_path}")

    elif cfg.mode == "spat":
        LOGGER.info("Generating Spatial 3D Scene Graph")
        # ============================================================
        # (0) Build spatial-only initial_graph for LLM
        # - drop "part" and "functional_relation"
        # - keep only top-level keys: "object", "spatial_relation"
        # ============================================================
        nd = 3  # rounding precision (same style as CA spatial input)
        graph_spat = {"object": {}, "spatial_relation": {}}

        if not isinstance(initial_graph, dict):
            LOGGER.error("[spat] initial_graph is not a dict. Check JSON load.")
            raise ValueError("[spat] initial_graph must be a dict")

        # copy prior spatial graph as-is (direction-agnostic candidates in prompt)
        graph_spat["spatial_relation"] = initial_graph.get("spatial_relation", {})

        # keep only object nodes (+ center/extent)
        if isinstance(initial_graph.get("object"), dict):
            for oid, obj in initial_graph["object"].items():
                if not (isinstance(oid, str) and oid.startswith("obj_")):
                    continue
                if not isinstance(obj, dict):
                    continue

                o = {"label": obj.get("label", "")}

                if isinstance(obj.get("center"), list):
                    try:
                        o["center"] = [round(float(x), nd) for x in obj["center"]]
                    except Exception:
                        o["center"] = obj.get("center")

                if isinstance(obj.get("extent"), list):
                    try:
                        o["extent"] = [round(float(x), nd) for x in obj["extent"]]
                    except Exception:
                        o["extent"] = obj.get("extent")

                graph_spat["object"][oid] = o

        graph_str_spat = json.dumps(graph_spat, ensure_ascii=False, separators=(",", ":"))
        LOGGER.info("[spat] Spatial-only initial_graph prepared (object + spatial_relation).")


        prompts = GPTprompt(config=cfg)
        LOGGER.info("Generating Spatial 3D Scene Graph without part nodes and affordance")
        LOGGER.info("API REQUESTING... (Spatial Relations)")
        resp = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=prompts.system_prompt_spatial,
            input=graph_str_spat,
            background=True,
            reasoning={"effort":"high"},
            text={
                "verbosity": "high",
                "format": {
                    "type": "json_schema",
                    "name": "Spatial3DSceneGraphEdits",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "replacements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pair": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern": "^obj_\\d+$"
                                            },
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        "old_label": {"type": "string"},
                                        "new_label": {
                                            "type": "string",
                                            "enum": [
                                                "near",
                                                "on",
                                                "with",
                                                "above",
                                                "in",
                                                "attached to",
                                                "has",
                                                "against"
                                            ]
                                        },
                                        "reason": {"type": "string"},
                                        "score": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0
                                        }
                                    },
                                    "required": ["pair", "old_label", "new_label", "reason", "score"],
                                    "additionalProperties": False
                                }
                            },
                            "removals": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pair": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern": "^obj_\\d+$"
                                            },
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        "old_label": {"type": "string"},
                                        "reason": {"type": "string"},
                                        "score": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0
                                        }
                                    },
                                    "required": ["pair", "old_label", "reason", "score"],
                                    "additionalProperties": False
                                }
                            },
                            "adds": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pair": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "pattern": "^obj_\\d+$"
                                            },
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        "label": {
                                            "type": "string",
                                            "enum": [
                                                "near",
                                                "on",
                                                "with",
                                                "above",
                                                "in",
                                                "attached to",
                                                "has",
                                                "against"
                                            ]
                                        },
                                        "reason": {"type": "string"},
                                        "score": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0
                                        }
                                    },
                                    "required": ["pair", "label", "reason", "score"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["replacements", "removals", "adds"],
                        "additionalProperties": False
                    }
                }
            }
        )

        # ============================================================
        # (2) Wait async job (same style as other modes)
        # ============================================================
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        timeout = 3000  # seconds
        start_time = time.time()
        prev_status = None

        while True:
            r = client.responses.retrieve(resp.id)
            s = getattr(r, "status", None)
            if s != prev_status:
                LOGGER.info(f"[spat] status changed: {prev_status} -> {s}")
                prev_status = s

            if s in {"completed", "succeeded"}:
                resp = r
                break
            if s in {"failed", "errored", "cancelled", "incomplete"}:
                LOGGER.error(f"[spat] job failed with status={s}")
                raise RuntimeError(f"[spat] job failed: {s}")

            if time.time() - start_time > timeout:
                LOGGER.error(f"[spat] job timeout after {timeout} seconds")
                raise TimeoutError(f"[spat] job timeout")

            sleep(5)

        # ============================================================
        # (3) Parse LLM edits JSON
        # ============================================================
        try:
            llm_edits = json.loads(resp.output_text)
            LOGGER.info("[spat] Successfully parsed LLM response as JSON - Spatial Edits")
        except Exception as e:
            LOGGER.error("[spat] Failed to parse LLM response as JSON.")
            LOGGER.error(f"[spat] Response: {resp}")
            raise e

        # ============================================================
        # (4) Save raw edits JSON (debug/trace)
        # ============================================================
        sg_dir = Path(cfg.dataset_root) / cfg.scene_id / cfg.save_folder_name / "scene_graph"
        sg_dir.mkdir(parents=True, exist_ok=True)

        edits_path = sg_dir / "spatial_3d_scene_graph_edits.json"
        with open(edits_path, "w") as jf:
            json.dump(llm_edits, jf, ensure_ascii=False, indent=2)
        LOGGER.info(f"[SceneGraph] Saved LLM edits: {edits_path}")

        # ============================================================
        # (5) Apply edits to prior spatial_relation and save refined graph
        # ============================================================
        obj_ids = set(graph_spat["object"].keys())

        def _norm_obj_id(x):
            if isinstance(x, str) and x.startswith("obj_"):
                return x
            if isinstance(x, str) and x.isdigit():
                return f"obj_{x}"
            return None

        def _edges_from_spatial_relation(sr):
            """
            Convert various possible prior formats into a canonical edge set:
            each edge is (a, b, label) where a/b are obj_ ids.

            Supported sr patterns:
            1) list of {"pair":[obj_i,obj_j], "label":"..."}   (direct)
            2) dict: obj_i -> {label: [obj_j, obj_k, ...]}    (label-grouped adjacency)
            3) dict: obj_i -> [{"obj": obj_j, "label": "..."}]
            4) dict: obj_i -> [obj_j, obj_k, ...]             (UNLABELED adjacency)
               -> default label="near" (best-effort)
            """
            edges = set()
            if sr is None:
                return edges

            # case 1: list-of-edges
            if isinstance(sr, list):
                for it in sr:
                    if not isinstance(it, dict):
                        continue
                    pair = it.get("pair", None)
                    lab = it.get("label", None)
                    if isinstance(pair, list) and len(pair) == 2:
                        a = _norm_obj_id(pair[0])
                        b = _norm_obj_id(pair[1])
                        if a and b and isinstance(lab, str) and lab:
                            edges.add((a, b, lab))
                return edges

            # case 2/3/4: adjacency dict
            if isinstance(sr, dict):
                for k, v in sr.items():
                    sub = _norm_obj_id(k)
                    if sub is None:
                        continue

                    # 2) label-grouped adjacency: obj_i -> {label: [obj_j,...]}
                    if isinstance(v, dict):
                        # if values look like list targets
                        for lab, targets in v.items():
                            if isinstance(targets, list):
                                for t in targets:
                                    obj = _norm_obj_id(t)
                                    if obj and obj != sub:
                                        # direction-agnostic candidate: store canonical order
                                        a, b = (sub, obj) if sub < obj else (obj, sub)
                                        edges.add((a, b, str(lab)))
                        # also support mapping: obj_i -> {obj_j: label}
                        if all(isinstance(val, str) for val in v.values()):
                            for tgt, lab in v.items():
                                obj = _norm_obj_id(tgt)
                                if obj and obj != sub:
                                    a, b = (sub, obj) if sub < obj else (obj, sub)
                                    edges.add((a, b, str(lab)))
                        continue

                    # 3) list of dicts OR 4) list of strings
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                tgt = _norm_obj_id(it.get("obj") or it.get("target") or it.get("id"))
                                lab = it.get("label")
                                if tgt and tgt != sub:
                                    a, b = (sub, tgt) if sub < tgt else (tgt, sub)
                                    if isinstance(lab, str) and lab:
                                        edges.add((a, b, lab))
                                    else:
                                        edges.add((a, b, "near"))
                            elif isinstance(it, str):
                                tgt = _norm_obj_id(it)
                                if tgt and tgt != sub:
                                    a, b = (sub, tgt) if sub < tgt else (tgt, sub)
                                    edges.add((a, b, "near"))
                return edges

            return edges

        def _apply_edits(prior_edges, edits):
            """
            prior_edges: set[(a,b,label)] where (a,b) might be canonical (undirected candidate)
            edits: {"replacements": [...], "removals": [...], "adds": [...]}

            - For matching removals/replacements: treat prior as direction-agnostic
              (try both orientations).
            - For adds/replacement outputs: use direction exactly as in edit["pair"].
            """
            edge_set = set(prior_edges)

            def _find_match(a, b, old_label):
                # direct match
                if (a, b, old_label) in edge_set:
                    return (a, b, old_label)
                if (b, a, old_label) in edge_set:
                    return (b, a, old_label)

                # fallback: match by pair ignoring label (only if unique)
                cand = [e for e in edge_set if (e[0] == a and e[1] == b) or (e[0] == b and e[1] == a)]
                if len(cand) == 1:
                    return cand[0]
                return None

            # removals
            for rem in edits.get("removals", []) if isinstance(edits, dict) else []:
                if not isinstance(rem, dict):
                    continue
                pair = rem.get("pair", [])
                old = rem.get("old_label", "")
                if not (isinstance(pair, list) and len(pair) == 2 and isinstance(old, str) and old):
                    continue
                a = _norm_obj_id(pair[0])
                b = _norm_obj_id(pair[1])
                if not (a and b) or a == b:
                    continue
                m = _find_match(a, b, old)
                if m is not None:
                    edge_set.discard(m)

            # replacements
            for rep in edits.get("replacements", []) if isinstance(edits, dict) else []:
                if not isinstance(rep, dict):
                    continue
                pair = rep.get("pair", [])
                old = rep.get("old_label", "")
                new = rep.get("new_label", "")
                if not (
                    isinstance(pair, list) and len(pair) == 2
                    and isinstance(old, str) and old
                    and isinstance(new, str) and new
                ):
                    continue
                a = _norm_obj_id(pair[0])
                b = _norm_obj_id(pair[1])
                if not (a and b) or a == b:
                    continue

                m = _find_match(a, b, old)
                if m is not None:
                    edge_set.discard(m)

                # add the directed replacement (pair order from LLM)
                edge_set.add((a, b, new))

            # adds
            for add in edits.get("adds", []) if isinstance(edits, dict) else []:
                if not isinstance(add, dict):
                    continue
                pair = add.get("pair", [])
                lab = add.get("label", "")
                if not (isinstance(pair, list) and len(pair) == 2 and isinstance(lab, str) and lab):
                    continue
                a = _norm_obj_id(pair[0])
                b = _norm_obj_id(pair[1])
                if not (a and b) or a == b:
                    continue
                edge_set.add((a, b, lab))

            return edge_set

        prior_edges = _edges_from_spatial_relation(graph_spat.get("spatial_relation"))
        refined_edges = _apply_edits(prior_edges, llm_edits)

        # Filter edges to existing object ids only
        refined_edges = {
            (a, b, lab) for (a, b, lab) in refined_edges
            if (a in obj_ids and b in obj_ids and isinstance(lab, str) and lab)
        }

        # Deterministic order
        def _edge_sort_key(e):
            a, b, lab = e
            ai = _idx_from(a, "obj")
            bi = _idx_from(b, "obj")
            ai = ai if ai is not None else 10**9
            bi = bi if bi is not None else 10**9
            return (ai, bi, lab)

        spatial_rel_out = [{"pair": [a, b], "label": lab} for (a, b, lab) in sorted(refined_edges, key=_edge_sort_key)]

        sg_out = {
            "object": graph_spat["object"],
            "spatial_relation": spatial_rel_out
        }

        sg_path = sg_dir / "spatial_3d_scene_graph.json"
        with open(sg_path, "w") as jf:
            json.dump(sg_out, jf, ensure_ascii=False, indent=2)
        LOGGER.info(f"[SceneGraph] Saved: {sg_path}")

    else:
        LOGGER.error(f"Unknown mode: {cfg.mode}")

    LOGGER.info("FINISH main()")

if __name__ == "__main__":
    main()