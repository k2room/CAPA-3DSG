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

    cfg.mode = "ca"   # ca: context-aware 3DSG, func: unified 3DSG-functional only

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
    elif ds == "CAPAD":
        cfg.dataset_root   = _resolve_path(cfg.CAPAD_root)
        cfg.dataset_config = _resolve_path(cfg.CAPAD_config)
        cfg.mode = "ca"
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
            instructions=prompts.system_func_local,
            input= initial_graph,
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
            instructions=prompts.system_func_remote,
            input= initial_graph,
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
        LOGGER.info("Generating Context-aware 3D Scene Graph with Spatial and Functional Relations")
        # TODO: implement full graph generation
        # Load scenario.json
        # Filtering obj/edge node based on scenarios
        # generate func_local, func_remote, spatial relations and affordance for each node
        scenario_path = (Path(cfg.dataset_root) / cfg.scene_id).parent / "scenario.json"
        if not scenario_path.exists():
            LOGGER.error(f"Scenario file not found: {scenario_path}")
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        scenario = read_json(scenario_path)
        LOGGER.info(f"Scenario loaded from: {scenario_path}")

        instruntions = []
        for i in scenario["scenario"].keys():
            sdata = scenario["scenario"][i]
            for j in sdata["route"].keys():
                instruntions.append([sdata["scenario_goal"], sdata["route"][j]["instruction"]])
        
        print(instruntions)
    
    
    
    
    else:
        LOGGER.error(f"Unknown mode: {cfg.mode}")

    LOGGER.info("FINISH main()")

if __name__ == "__main__":
    main()