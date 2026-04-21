#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenIE-PT v11 BIA theoretical experiment (self-contained)

Objetivo:
- extrair triplas o mais próximo possível do BIA;
- registrar padrões UD (gold e predições);
- usar BERTimbau para rankear/inspecionar cabeças de atenção;
- opcionalmente filtrar candidatos por score de atenção;
- gerar saídas completas para análise em notebook.

Saídas principais:
- predictions_enriched.jsonl / .csv
- gold_enriched.csv
- metrics.json
- selected_heads.json
- triples_table.csv
"""

from __future__ import annotations

import pandas as pd
import argparse
import csv
import gzip
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import stanza
from conllu import parse_incr
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("openie_v11_bia_theoretical_selfcontained")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    bert_model: str = "neuralmind/bert-base-portuguese-cased"
    lang: str = "pt"
    use_gpu_stanza: bool = False
    max_tokens_bert: int = 256
    seed: int = 13

    # extração
    deduplicate_by_verb: bool = True
    generate_obl_variants: bool = True
    obl_deps: Set[str] = field(default_factory=lambda: {"obl", "obl:arg"})
    extract_coord: bool = True
    extract_aux_pass: bool = True
    extract_cop: bool = True
    extract_modal: bool = True
    extract_conj_verbs: bool = True

    # viés para aproximar BIA
    prefer_longer_rel_copula: bool = True
    keep_simple_copula_variant: bool = True
    keep_numeric_args: bool = True
    disable_appos_alias: bool = True
    disable_zero_copula_name: bool = True

    # matching
    use_prefix_match: bool = True
    prefix_min_tokens: int = 1
    use_partial_rel: bool = True
    use_substring_match: bool = True

    # atenção
    attn_threshold: float = 0.0
    heads_mode: str = "rank"  # rank | all | forced
    top_k_heads: int = 10
    n_sent_rank: int = 300
    window_s: Tuple[int, int] = (2, 5)
    window_o: Tuple[int, int] = (7, 12)
    bosque_path: Optional[str] = None
    forced_heads_s: Optional[Union[str, List[Tuple[int, int]], List[List[int]]]] = None
    forced_heads_o: Optional[Union[str, List[Tuple[int, int]], List[List[int]]]] = None
    no_attn: bool = False
    allow_fallback_all_heads: bool = False
    cop_mode: str = "full"  # full | off | restricted
    random_heads_seed: int = 13
    attn_decision_enabled: bool = True
    attn_decision_patterns: Tuple[str, ...] = ("verb+obj", "verb+obl", "cop", "aux:pass")
    attn_keep_if_missing: bool = True
    attn_filter_in_legacy: bool = False

    attn_rerank_enabled: bool = True
    attn_rerank_patterns: Tuple[str, ...] = ("verb+obj", "verb+obl", "cop", "aux:pass")
    attn_rerank_keep_top1: bool = True
    attn_rerank_weight: float = 1.0


# -----------------------------------------------------------------------------
# Constantes linguísticas
# -----------------------------------------------------------------------------
CLAUSE_SKIP: Set[str] = {"acl", "acl:relcl"}
INVALID_SUBJ_LEMMAS: Set[str] = {"que", "quem", "qual", "cujo", "cuja", "cujos", "cujas"}
ANAPHORIC_PRON: Set[str] = {"ele", "ela", "eles", "elas", "isso", "isto"}
CLITIC_OBJ: Set[str] = {"a", "o", "as", "os", "lhe", "lhes", "me", "te", "se", "nos", "vos"}
COPULA_VERBS: Set[str] = {"ser", "estar", "ficar", "parecer", "permanecer", "continuar", "tornar"}
PREPS: Set[str] = {"de", "a", "em", "para", "com", "por", "sobre", "até", "entre", "sem", "sob"}
MODAIS: Set[str] = {"poder", "dever", "conseguir", "querer", "precisar", "tentar"}
GOLD_REL_SKIP: Set[str] = {
    "também", "ainda", "já", "só", "somente", "apenas", "mesmo", "até",
    "normalmente", "geralmente", "quase", "praticamente",
}
DET_START: Set[str] = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "este", "esta", "esse", "essa", "aquele", "aquela",
    "meu", "minha", "seu", "sua", "nosso", "nossa",
}
ADVMOD_EXCLUDE: Set[str] = {
    "talvez", "provavelmente", "possivelmente", "certamente", "realmente",
    "sempre", "nunca", "jamais", "hoje", "ontem", "amanhã", "agora", "ainda",
    "já", "logo", "então", "frequentemente", "raramente", "geralmente",
    "apenas", "só", "somente", "também", "muito", "pouco", "bem", "mal",
    "mais", "menos", "tão", "quão", "até", "mesmo", "aqui", "lá", "ali", "aí", "cá",
    "não", "nao", "nem", "normalmente", "quase",
    "quando", "se", "enquanto", "porque", "embora", "caso", "apesar",
}
CONTRACTION_MAP: Dict[str, str] = {
    "do": "de o", "da": "de a", "dos": "de os", "das": "de as",
    "ao": "a o", "aos": "a os", "à": "a a", "às": "a as",
    "no": "em o", "na": "em a", "nos": "em os", "nas": "em as",
    "pelo": "por o", "pela": "por a", "pelos": "por os", "pelas": "por as",
    "num": "em um", "numa": "em uma", "dum": "de um", "duma": "de uma",
    "deste": "de este", "desta": "de esta", "neste": "em este", "nesta": "em esta",
    "desse": "de esse", "dessa": "de essa", "nesse": "em esse", "nessa": "em essa",
    "daquele": "de aquele", "daquela": "de aquela", "naquele": "em aquele",
}


@dataclass
class UDTok:
    id: int
    text: str
    lemma: str
    upos: str
    head: int
    deprel: str
    start_char: int
    end_char: int


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def expand_contractions(text: str) -> str:
    return " ".join(CONTRACTION_MAP.get(w.lower(), w) for w in (text or "").split())


_RX_WS = re.compile(r"\s+")
_RX_PUNCT = re.compile(r"[^\w\sÀ-ÿ£%\-]", flags=re.UNICODE)


def _norm(s: str) -> str:
    s = expand_contractions((s or "").strip().lower())
    s = _RX_PUNCT.sub(" ", s)
    return _RX_WS.sub(" ", s).strip()


def canon_arg(s: str) -> str:
    toks = _norm(s).split()
    while toks and toks[0] in DET_START:
        toks = toks[1:]
    return " ".join(toks)


def canon_rel(s: str) -> str:
    return _norm(s)


def parse_window(s: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if not s:
        return default
    s = s.replace("--", ":").replace("-", ":")
    parts = [p.strip() for p in s.split(":") if p.strip()]
    if len(parts) != 2:
        return default
    return int(parts[0]), int(parts[1])


def parse_forced_heads(x: Union[str, List[Tuple[int, int]], List[List[int]], None]) -> List[Tuple[int, int]]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for p in x:
            if p is None:
                continue
            if isinstance(p, (tuple, list)) and len(p) == 2:
                out.append((int(p[0]), int(p[1])))
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        out = []
        for chunk in s.split(","):
            chunk = chunk.strip()
            if "-" in chunk:
                a, b = chunk.split("-", 1)
            elif ":" in chunk:
                a, b = chunk.split(":", 1)
            else:
                continue
            try:
                out.append((int(a.strip()), int(b.strip())))
            except Exception:
                pass
        return out
    raise TypeError(f"forced_heads inválido: {type(x)}")


def validate_heads(heads: Iterable[Tuple[int, int]], n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    out = []
    for l, h in heads:
        if 1 <= l <= n_layers and 1 <= h <= n_heads:
            out.append((int(l), int(h)))
    return out


def read_gold_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            sent = obj.get("sentence", "")
            golds = obj.get("gold", [])
            if isinstance(golds, dict):
                golds = [golds]
            rows.append({
                "sentence": sent,
                "gold": [
                    {
                        "arg1": g.get("arg1", ""),
                        "rel": g.get("rel", ""),
                        "arg2": g.get("arg2", ""),
                        "valid": g.get("valid", True),
                    }
                    for g in golds
                ],
                "doc_id": obj.get("doc_id"),
                "phrase_index": obj.get("phrase_index"),
            })
    return rows


def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Union[str, Path], rows: List[Dict[str, Any]], delimiter: str = ";"):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def jaccard_accuracy(tp: int, fp: int, fn: int) -> float:
    den = tp + fp + fn
    return tp / den if den else 0.0


# -----------------------------------------------------------------------------
# Atenção / ranking de heads
# -----------------------------------------------------------------------------
def _calc_offsets_in_text(text: str, toks: List[Dict[str, Any]]) -> None:
    i = 0
    for t in toks:
        form = t["form"]
        pos = text.find(form, i)
        if pos < 0:
            pos = i
        t["start_char"] = pos
        t["end_char"] = pos + len(form)
        i = t["end_char"]


def _map_tokens_to_wp_simple(text: str, toks: List[Dict[str, Any]], tokenizer) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=256)
    offsets = enc["offset_mapping"]
    mapping: Dict[int, List[int]] = {int(t["id"]): [] for t in toks}
    for wp_idx, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            continue
        for t in toks:
            if not (t["end_char"] <= a or b <= t["start_char"]):
                mapping[int(t["id"])].append(wp_idx)
    return mapping, enc


def attn_score_pair(M: torch.Tensor, src_wp: List[int], tgt_wp: List[int]) -> float:
    if not src_wp or not tgt_wp:
        return 0.0
    sub = M[src_wp][:, tgt_wp]
    return float(sub.mean().item())


def average_selected_heads(attentions: List[torch.Tensor], selected_heads: List[Tuple[int, int]], layer_window: Tuple[int, int]) -> torch.Tensor:
    l1, l2 = layer_window
    mats = []
    for (l, h) in selected_heads:
        if l1 <= l <= l2:
            mats.append(attentions[l - 1][0, h - 1, :, :])
    if not mats:
        for l in range(l1, l2 + 1):
            mats.append(attentions[l - 1][0].mean(0))
    return torch.stack(mats, 0).mean(0)


def load_bosque(path: str, max_sent: int = 300) -> List[Tuple[str, List[Dict[str, Any]]]]:
    data = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        count = 0
        for sent in parse_incr(fh):
            text = sent.metadata.get("text", None) or " ".join(tok["form"] for tok in sent)
            toks = []
            for tok in sent:
                if isinstance(tok["id"], tuple):
                    continue
                toks.append(tok)
            data.append((text, toks))
            count += 1
            if count >= max_sent:
                break
    return data


def bosque_heads_gain_roleaware(tokenizer, model, bosque_pairs, n_layers, n_heads, window, role, device=None):
    l1, l2 = window
    scores = torch.zeros((n_layers, n_heads), dtype=torch.float32, device=device)
    counts = torch.zeros((n_layers, n_heads), dtype=torch.float32, device=device) + 1e-8
    foldable_preps = {"em", "de", "a", "para", "por", "como"}

    for (text, toks_raw) in tqdm(bosque_pairs, desc=f"Ranking heads/{role}", leave=False):
        toks = [t for t in toks_raw if not isinstance(t["id"], tuple)]
        _calc_offsets_in_text(text, toks)
        kids: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in toks:
            hid = int(t.get("head") or 0)
            kids[hid].append(t)
        mapping, _ = _map_tokens_to_wp_simple(text, toks, tokenizer)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)
        atts = out.attentions

        for t in toks:
            if (t.get("upos") or "").upper() != "VERB":
                continue
            v_id = int(t["id"])
            v_wp = mapping.get(v_id, [])
            if not v_wp:
                continue
            dep_nodes: List[Dict[str, Any]] = []
            if role == "subj":
                dep_nodes = [c for c in kids.get(v_id, []) if (c.get("deprel") or "").startswith(("nsubj", "csubj"))]
            elif role == "obj":
                dep_nodes = [c for c in kids.get(v_id, []) if c.get("deprel") in {"obj", "iobj"}]
                for c in kids.get(v_id, []):
                    if c.get("deprel") == "obl":
                        if any((ch.get("deprel") == "case" and (ch.get("lemma") or ch.get("form")).lower() in foldable_preps)
                               for ch in kids.get(int(c["id"]), [])):
                            dep_nodes.append(c)
            for dep in dep_nodes:
                d_wp = mapping.get(int(dep["id"]), [])
                if not d_wp:
                    continue
                for l in range(max(l1, 1) - 1, min(l2, n_layers)):
                    M = atts[l][0]
                    gmean = M.mean().item()
                    for h in range(n_heads):
                        sub = M[h][:, :][v_wp][:, d_wp].mean().item()
                        scores[l, h] += (sub - gmean)
                        counts[l, h] += 1.0

    mean_score = scores / counts
    triples = []
    cpu = mean_score.detach().cpu()
    for l in range(n_layers):
        for h in range(n_heads):
            triples.append((l + 1, h + 1, float(cpu[l, h].item())))
    triples = [t for t in triples if l1 <= t[0] <= l2]
    triples.sort(key=lambda x: x[2], reverse=True)
    return triples


# -----------------------------------------------------------------------------
# Extrator
# -----------------------------------------------------------------------------
class OpenIEExtractorBIA:
    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.config = config or Config()
        self.verbose = verbose
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_stanza()
        self._init_bert()
        self._select_heads()
        if self.verbose:
            logger.info("✅ OpenIE-PT BIA-aligned inicializado")

    def _pattern_uses_attention_rerank(self, pattern_ud: str) -> bool:
        if self.config.no_attn:
            return False
        if not getattr(self.config, "attn_rerank_enabled", False):
            return False
        allowed = set(getattr(self.config, "attn_rerank_patterns", ()))
        return pattern_ud in allowed

    def _rerank_score(self, row: Dict[str, Any]) -> float:
        """
        Score simples de reranking.
        Hoje usa principalmente conf_att, mas pode ser expandido depois.
        """
        conf = row.get("conf_att", None)
        try:
            conf_val = float(conf) if conf is not None else 0.0
        except Exception:
            conf_val = 0.0
    
        base = conf_val * float(getattr(self.config, "attn_rerank_weight", 1.0))
    
        # pequeno bônus para theory_valid
        if row.get("theory_valid", False):
            base += 0.001
    
        return base

    def _rerank_group_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        Define quais candidatos competem entre si.
        """
        return (
            row.get("sentence", ""),
            row.get("verb_id"),
            row.get("pattern_ud", "unknown"),
        )

    def _apply_attention_rerank(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Mantém apenas o melhor candidato por grupo de competição
        para padrões onde o reranking por atenção estiver habilitado.
        """
        if self.config.no_attn or not getattr(self.config, "attn_rerank_enabled", False):
            return rows
    
        keep = []
        grouped = {}
    
        for row in rows:
            pattern = row.get("pattern_ud", "unknown")
    
            # padrões fora do rerank ficam como estão
            if not self._pattern_uses_attention_rerank(pattern):
                keep.append(row)
                continue
    
            gk = self._rerank_group_key(row)
            score = self._rerank_score(row)
    
            prev = grouped.get(gk)
            if prev is None or score > prev["_rerank_score"]:
                row = dict(row)
                row["_rerank_score"] = score
                grouped[gk] = row
    
        reranked = keep + list(grouped.values())
    
        for row in reranked:
            row.pop("_rerank_score", None)
    
        return reranked

    def _pattern_uses_attention_decision(self, pattern_ud: str) -> bool:
        if self.config.no_attn:
            return False
        if not getattr(self.config, "attn_decision_enabled", True):
            return False
        allowed = set(getattr(self.config, "attn_decision_patterns", ()))
        return pattern_ud in allowed

    def _passes_attention_gate(self, row: Dict[str, Any]) -> bool:
        """
        Aplica threshold de atenção somente em padrões selecionados.
        """
        if self.config.no_attn:
            return True
    
        pattern = row.get("pattern_ud", "unknown")
        if not self._pattern_uses_attention_decision(pattern):
            return True
    
        conf = row.get("conf_att", None)
    
        # se não houver atenção calculada, mantém ou remove conforme config
        if conf is None:
            return bool(getattr(self.config, "attn_keep_if_missing", True))
    
        try:
            return float(conf) >= float(self.config.attn_threshold)
        except Exception:
            return bool(getattr(self.config, "attn_keep_if_missing", True))
    
    def _truncate_legacy_arg_text(self, text: str) -> str:
        """
        Limpa spans já textualizados vindos do extrator legado.
        Corta em parentético, glosa metalinguística e aposto explicativo comum.
        """
        if not text:
            return text
    
        s = text.strip()
    
        # corta em parentético
        s = re.split(r"\s*\(", s, maxsplit=1)[0].strip()
    
        # corta em glosas metalinguísticas
        s = re.split(r"\b(em\s+latim|em\s+inglês|em\s+português|em\s+francês|em\s+espanhol)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    
        # corta em vírgula explicativa tardia
        s = re.split(r"\s*,\s*", s, maxsplit=1)[0].strip()

        return s

    def _token_index(self, tok_id: Optional[int], ud: List[UDTok]) -> int:
        if tok_id is None:
            return -1
        for i, t in enumerate(ud):
            if t.id == tok_id:
                return i
        return -1


    def _is_inside_parenthetical_window(self, tok_id: Optional[int], ud: List[UDTok], window: int = 8) -> bool:
        idx = self._token_index(tok_id, ud)
        if idx < 0:
            return False
        left = " ".join(t.text for t in ud[max(0, idx - window):idx])
        right = " ".join(t.text for t in ud[idx:min(len(ud), idx + window + 1)])
        ctx = f"{left} {right}"
        return "(" in ctx or ")" in ctx
    
    
    def _arg_span_text_from_head(self, head_id: int, ud: List[UDTok], kids: Dict[int, List[UDTok]], extensive: bool = True) -> str:
        span_ids = self._expand_arg_extensive(head_id, ud, kids) if extensive else self._expand_arg_minimal(head_id, ud, kids)
        return canon_arg(self._span_text(span_ids, ud))
    
    
    def _obl_case_text(self, obl_tok: UDTok, kids: Dict[int, List[UDTok]]) -> str:
        case_children = [c for c in kids.get(obl_tok.id, []) if (c.deprel or "").startswith("case")]
        return canon_arg(" ".join(c.text for c in sorted(case_children, key=lambda x: x.id)))
    
    
    def _is_temporal_like_obl(self, obl_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        span = self._arg_span_text_from_head(obl_tok.id, ud, kids, extensive=True)
        case_txt = self._obl_case_text(obl_tok, kids)
    
        temporal_patterns = [
            r"\b\d{4}\b",
            r"\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b",
            r"\b(ontem|hoje|amanhã|antes|depois|durante)\b",
            r"\b(em|no|na)\s+\d{4}\b",
        ]
        if any(re.search(p, span) for p in temporal_patterns):
            return True
    
        if case_txt in {"em", "no", "na"} and re.search(r"\b\d{4}\b", span):
            return True
    
        return False
    
    
    def _is_metalinguistic_obl(self, obl_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        span = self._arg_span_text_from_head(obl_tok.id, ud, kids, extensive=True)
        bad_phrases = {
            "em latim", "em inglês", "em português", "em francês", "em espanhol",
            "abreviado", "abreviada", "abreviados", "abreviaturas",
            "pós nominais", "pos nominais", "sigla", "siglas",
            "literalmente", "traduzido", "tradução",
        }
        return any(bp in span for bp in bad_phrases)
    
    
    def _is_discourse_or_intro_obl(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if obl_tok.id > verb_tok.id:
            return False
    
        span = self._arg_span_text_from_head(obl_tok.id, ud, kids, extensive=True)
        case_txt = self._obl_case_text(obl_tok, kids)
    
        intro_markers = {"em", "durante", "após", "antes", "desde", "sob", "entre"}
        if case_txt in intro_markers:
            return True
    
        discourse_patterns = [
            r"^em \d{4}\b",
            r"^em fevereiro\b",
            r"^em março\b",
            r"^em abril\b",
            r"^em maio\b",
            r"^em junho\b",
            r"^em julho\b",
            r"^em agosto\b",
            r"^em setembro\b",
            r"^em outubro\b",
            r"^em novembro\b",
            r"^em dezembro\b",
        ]
        return any(re.search(p, span) for p in discourse_patterns)
    
    
    def _is_location_like_obl(self, obl_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        span = self._arg_span_text_from_head(obl_tok.id, ud, kids, extensive=True)
        case_txt = self._obl_case_text(obl_tok, kids)
    
        if case_txt in {"em", "no", "na", "nos", "nas", "para", "ao", "a"}:
            if re.search(r"\b([a-zà-ÿ]+)\b", span):
                return True
        return False
    
    
    def _obl_semantic_class(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> str:
        if self._is_inside_parenthetical_window(obl_tok.id, ud):
            return "parenthetical"
    
        if self._is_metalinguistic_obl(obl_tok, ud, kids):
            return "metalinguistic"
    
        # temporal e introdutório precisam vir antes de locativo
        if self._is_discourse_or_intro_obl(obl_tok, verb_tok, ud, kids):
            return "intro"
    
        if self._is_temporal_like_obl(obl_tok, ud, kids):
            return "temporal"
    
        if self._is_location_like_obl(obl_tok, ud, kids):
            return "locative"
    
        return "other"
    
    
    def _obl_is_core_for_attachment(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        Decide se o obl pode ser anexado como complemento central.
        Para base_verb_obl, seja bem restritivo.
        """
        cls = self._obl_semantic_class(obl_tok, verb_tok, ud, kids)
    
        # somente locativos/destino realmente centrais
        if cls != "locative":
            return False
    
        span = self._arg_span_text_from_head(obl_tok.id, ud, kids, extensive=True)
    
        # bloqueia locativos claramente temporais disfarçados
        if re.search(r"\b\d{4}\b", span):
            return False
        if re.search(r"\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b", span):
            return False
    
        return True
    
    
    def _obl_is_core_for_split(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        Decide se o obl justifica decomposição n-ária em tripla própria.
        Também restritivo, mas permite alguns temporais centrais.
        """
        cls = self._obl_semantic_class(obl_tok, verb_tok, ud, kids)
        return cls in {"locative", "temporal"}

    def _init_stanza(self):
        if self.verbose:
            logger.info("🔄 Carregando Stanza...")
        try:
            self.nlp = stanza.Pipeline(
                lang=self.config.lang,
                processors="tokenize,mwt,pos,lemma,depparse",
                tokenize_no_ssplit=True,
                verbose=False,
                use_gpu=self.config.use_gpu_stanza and torch.cuda.is_available(),
                download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            )
        except Exception:
            stanza.download(self.config.lang, verbose=False)
            self.nlp = stanza.Pipeline(
                lang=self.config.lang,
                processors="tokenize,mwt,pos,lemma,depparse",
                tokenize_no_ssplit=True,
                verbose=False,
                use_gpu=self.config.use_gpu_stanza and torch.cuda.is_available(),
            )

    def _init_bert(self):
        if self.config.no_attn:
            self.tokenizer = None
            self.model = None
            self.heads_s = []
            self.heads_o = []
            return
        if self.verbose:
            logger.info(f"🔄 Carregando {self.config.bert_model} em {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model, use_fast=True)
        try:
            self.model = AutoModel.from_pretrained(
                self.config.bert_model,
                output_attentions=True,
                attn_implementation="eager",
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(self.config.bert_model, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()

    def _select_heads(self):
        if self.config.no_attn:
            self.heads_s, self.heads_o = [], []
            self.heads_meta = {"mode": "none", "heads_s": [], "heads_o": [], "head_analysis_valid": False}
            return

        n_layers = self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads

        def all_heads():
            return [(l, h) for l in range(1, n_layers + 1) for h in range(1, n_heads + 1)]

        mode = (self.config.heads_mode or "rank").lower()
        if mode == "all":
            hs = all_heads()
            ho = list(hs)
            meta = {"mode": "all", "heads_s": hs, "heads_o": ho, "gains_s": [], "gains_o": [], "head_analysis_valid": False}
        elif mode == "random":
            ah = all_heads()
            rng = random.Random(self.config.random_heads_seed)
            k = max(1, min(self.config.top_k_heads, len(ah)))
            hs = sorted(rng.sample(ah, k))
            ho = sorted(rng.sample(ah, k))
            meta = {"mode": "random", "heads_s": hs, "heads_o": ho, "gains_s": [], "gains_o": [], "head_analysis_valid": False, "random_heads_seed": self.config.random_heads_seed}
        elif mode == "forced":
            hs = validate_heads(parse_forced_heads(self.config.forced_heads_s), n_layers, n_heads)
            ho = validate_heads(parse_forced_heads(self.config.forced_heads_o), n_layers, n_heads)
            if not hs or not ho:
                raise ValueError("heads_mode='forced' exige forced_heads_s e forced_heads_o válidos")
            meta = {"mode": "forced", "heads_s": hs, "heads_o": ho, "gains_s": [], "gains_o": [], "head_analysis_valid": True}
        else:
            if not self.config.bosque_path:
                if self.config.allow_fallback_all_heads:
                    logger.warning("[WARN] bosque_path ausente em mode=rank; usando todas as heads como fallback")
                    hs = all_heads()
                    ho = list(hs)
                    meta = {"mode": "all_fallback", "heads_s": hs, "heads_o": ho, "gains_s": [], "gains_o": [], "head_analysis_valid": False}
                else:
                    raise FileNotFoundError("heads_mode='rank' exige --bosque. Sem Bosque não há análise válida de cabeças.")
            else:
                bosque = Path(self.config.bosque_path)
                if not bosque.exists() or not bosque.is_file():
                    if self.config.allow_fallback_all_heads:
                        logger.warning(f"[WARN] Bosque não encontrado ({bosque}); usando fallback all heads")
                        hs = all_heads()
                        ho = list(hs)
                        meta = {"mode": "all_fallback", "heads_s": hs, "heads_o": ho, "gains_s": [], "gains_o": [], "head_analysis_valid": False}
                    else:
                        raise FileNotFoundError(f"Bosque não encontrado: {bosque.resolve()}")
                else:
                    pairs = load_bosque(self.config.bosque_path, max_sent=self.config.n_sent_rank)
                    gains_s = bosque_heads_gain_roleaware(
                        self.tokenizer, self.model, pairs,
                        n_layers, n_heads, self.config.window_s, "subj", self.device
                    )
                    gains_o = bosque_heads_gain_roleaware(
                        self.tokenizer, self.model, pairs,
                        n_layers, n_heads, self.config.window_o, "obj", self.device
                    )
                    hs = [(l, h) for (l, h, _) in gains_s[: self.config.top_k_heads]]
                    ho = [(l, h) for (l, h, _) in gains_o[: self.config.top_k_heads]]
                    if not hs or not ho:
                        raise RuntimeError("Ranking retornou vazio para sujeito/objeto. Revise Bosque, janelas e top_k.")
                    meta = {
                        "mode": "rank",
                        "heads_s": hs,
                        "heads_o": ho,
                        "gains_s": [{"layer": l, "head": h, "gain": g} for (l, h, g) in gains_s[: self.config.top_k_heads]],
                        "gains_o": [{"layer": l, "head": h, "gain": g} for (l, h, g) in gains_o[: self.config.top_k_heads]],
                        "head_analysis_valid": True,
                    }
        self.heads_s, self.heads_o = hs, ho
        self.heads_meta = meta

    # ----------------------------- parse UD -----------------------------
    def _parse_ud(self, sentence: str) -> List[UDTok]:
        doc = self.nlp(sentence)
        s = doc.sentences[0]
        return [
            UDTok(
                id=int(w.id), text=w.text, lemma=(w.lemma or w.text), upos=w.upos,
                head=int(w.head), deprel=w.deprel,
                start_char=int(w.start_char) if w.start_char is not None else -1,
                end_char=int(w.end_char) if w.end_char is not None else -1,
            )
            for w in s.words
        ]

    def _children_index(self, toks: List[UDTok]) -> Dict[int, List[UDTok]]:
        kids: Dict[int, List[UDTok]] = defaultdict(list)
        for t in toks:
            if t.head and t.head != 0:
                kids[t.head].append(t)
        return kids

    def _span_text(self, span_ids: List[int], toks: List[UDTok]) -> str:
        id2tok = {t.id: t for t in toks}
        raw = normalize_space(" ".join(id2tok[i].text for i in sorted(span_ids) if i in id2tok))
        return expand_contractions(raw)

    def _expand_arg(self, head_id: int, toks: List[UDTok], kids: Dict[int, List[UDTok]], allow_appos: bool = True,
                    allow_conj: bool = True, exclude_ids: Optional[Set[int]] = None) -> List[int]:
        exclude_ids = exclude_ids or set()
        span = {head_id}
        agenda = [head_id]
        visited = {head_id}
        allowed = {"det", "amod", "nummod", "compound", "compound:prt", "fixed", "nmod", "nmod:poss", "case", "clf", "flat", "flat:name", "appos" if allow_appos else "__no__"}
        if allow_conj:
            allowed |= {"conj", "cc"}
        while agenda:
            x = agenda.pop()
            for c in kids.get(x, []):
                if c.deprel in allowed and c.id not in visited and c.id not in exclude_ids:
                    span.add(c.id)
                    visited.add(c.id)
                    agenda.append(c.id)
        return sorted(span)

    def _expand_subj(self, head_id: int, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[int]:
        return self._expand_arg(head_id, toks, kids, allow_appos=False, allow_conj=True)

    def _expand_obj(self, head_id: int, toks: List[UDTok], kids: Dict[int, List[UDTok]], allow_conj: bool = True,
                    exclude_ids: Optional[Set[int]] = None) -> List[int]:
        return self._expand_arg(head_id, toks, kids, allow_appos=True, allow_conj=allow_conj, exclude_ids=exclude_ids)

    def _is_invalid_subj(self, tok: UDTok) -> bool:
        return (tok.lemma or tok.text).lower() in INVALID_SUBJ_LEMMAS or tok.text.lower() in ANAPHORIC_PRON

    def _is_clitic_obj(self, tok: UDTok, span: List[int]) -> bool:
        return len(span) == 1 and (tok.text or "").lower() in CLITIC_OBJ

    def _has_double_prep(self, rel_str: str, arg2_str: str) -> bool:
        rel_c = normalize_space(expand_contractions(rel_str.lower()))
        arg2_c = normalize_space(expand_contractions(arg2_str.lower()))
        arg2_words = arg2_c.split()
        if not arg2_words:
            return False
        first = arg2_words[0]
        if first not in PREPS or first == "a":
            return False
        return first in set(rel_c.split())

    def _build_rel(self, v: UDTok, v_kids: List[UDTok], obj_head_id: int, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> Tuple[str, Set[int], Set[int]]:
        id2tok = {t.id: t for t in toks}
        rel_ids: Set[int] = {v.id}
        prep_excl: Set[int] = set()
        for c in v_kids:
            if c.deprel in {"aux", "aux:pass", "compound:prt", "fixed"}:
                rel_ids.add(c.id)
            if c.deprel in {"expl", "expl:pv"} and c.upos == "PRON":
                rel_ids.add(c.id)
            if c.deprel == "advmod" and (c.lemma or c.text).lower() in {"não", "nao", "nem", "nunca", "jamais"}:
                rel_ids.add(c.id)
        obj_kids = kids.get(obj_head_id, [])
        for c in obj_kids:
            if c.deprel == "case" and c.id < obj_head_id:
                between = [t for t in toks if v.id < t.id < c.id and t.upos not in {"PUNCT", "SYM"}]
                all_in_rel = all(t.id in rel_ids or t.deprel in {"aux", "aux:pass"} for t in between)
                if not between or all_in_rel:
                    rel_ids.add(c.id)
                    prep_excl.add(c.id)
                break
        rel_raw = normalize_space(" ".join(id2tok[i].text for i in sorted(rel_ids) if i in id2tok))
        return expand_contractions(rel_raw) or v.text, rel_ids, prep_excl

    # ----------------------------- extração -----------------------------
    def _extract_coordinated_elements(self, head: UDTok, kids: Dict[int, List[UDTok]], toks: List[UDTok]) -> List[List[int]]:
        conj_heads = [head]
        for c in kids.get(head.id, []):
            if c.deprel == "conj":
                conj_heads.append(c)
        if len(conj_heads) <= 1:
            return []
        result = []
        for h in conj_heads:
            span = self._expand_arg(h.id, toks, kids, allow_conj=False, allow_appos=True)
            result.append(span)
        return result

    def _extract_coordinated_verbs(self, sentence: str, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[Dict[str, Any]]:
        triplas = []
        id2tok = {t.id: t for t in toks}
        for v in toks:
            if v.upos not in {"VERB", "AUX"} or v.deprel != "conj":
                continue
            parent = id2tok.get(v.head)
            if not parent or parent.upos not in {"VERB", "AUX"}:
                continue
            v_kids = kids.get(v.id, [])
            obj = next((c for c in v_kids if c.deprel == "obj" or c.deprel.startswith("obj:")), None)
            if not obj:
                continue
            subj = next((c for c in v_kids if c.deprel in {"nsubj", "nsubj:pass"}), None)
            if not subj:
                parent_kids = kids.get(parent.id, [])
                subj = next((c for c in parent_kids if c.deprel in {"nsubj", "nsubj:pass"}), None)
            if not subj or self._is_invalid_subj(subj):
                continue
            rel, _, prep_excl = self._build_rel(v, v_kids, obj.id, toks, kids)
            subj_span = self._expand_subj(subj.id, toks, kids)
            obj_span = self._expand_obj(obj.id, toks, kids, allow_conj=True, exclude_ids=prep_excl)
            if self._is_clitic_obj(obj, obj_span):
                continue
            arg1 = self._span_text(subj_span, toks)
            arg2 = self._span_text(obj_span, toks)
            if arg1 and arg2 and rel and not self._has_double_prep(rel, arg2):
                triplas.append({
                    "sentence": sentence, "arg1": arg1, "rel": rel, "arg2": arg2,
                    "verb_id": v.id, "subj_id": subj.id, "obj_id": obj.id,
                    "pattern_ud": "verb+obj[conj]", "subj_deprel": subj.deprel, "obj_deprel": obj.deprel,
                })
        return triplas

    def _extract_modal(self, sentence: str, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[Dict[str, Any]]:
        triplas = []
        id2tok = {t.id: t for t in toks}
        for v in toks:
            if v.upos not in {"VERB", "AUX"}:
                continue
            if (v.lemma or v.text).lower() not in MODAIS:
                continue
            v_kids = kids.get(v.id, [])
            main_verb = next((c for c in v_kids if c.deprel == "xcomp" and c.upos == "VERB"), None)
            if not main_verb:
                continue
            main_kids = kids.get(main_verb.id, [])
            subj = next((c for c in v_kids if c.deprel in {"nsubj", "nsubj:pass"}), None) or \
                   next((c for c in main_kids if c.deprel in {"nsubj", "nsubj:pass"}), None)
            if not subj or self._is_invalid_subj(subj):
                continue
            obj = next((c for c in main_kids if c.deprel == "obj" or c.deprel.startswith("obj:")), None)
            if not obj:
                obj = next((c for c in main_kids if c.deprel in {"obl", "obl:arg"}), None)
            if not obj:
                continue
            rel_ids = {v.id, main_verb.id}
            obj_kids = kids.get(obj.id, [])
            case_tok = next((c for c in obj_kids if c.deprel == "case"), None)
            prep_excl = set()
            if case_tok:
                rel_ids.add(case_tok.id)
                prep_excl.add(case_tok.id)
            rel_raw = " ".join(id2tok[i].text for i in sorted(rel_ids) if i in id2tok)
            rel = expand_contractions(normalize_space(rel_raw))
            subj_span = self._expand_subj(subj.id, toks, kids)
            obj_span = self._expand_obj(obj.id, toks, kids, allow_conj=True, exclude_ids=prep_excl)
            arg1 = self._span_text(subj_span, toks)
            arg2 = self._span_text(obj_span, toks)
            if arg1 and arg2 and rel and not self._has_double_prep(rel, arg2):
                triplas.append({
                    "sentence": sentence, "arg1": arg1, "rel": rel, "arg2": arg2,
                    "verb_id": v.id, "subj_id": subj.id, "obj_id": obj.id,
                    "pattern_ud": "modal+verb", "subj_deprel": subj.deprel, "obj_deprel": obj.deprel,
                })
        return triplas

    def _extract_aux_pass(self, sentence: str, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[Dict[str, Any]]:
        triplas = []
        id2tok = {t.id: t for t in toks}
        for v in toks:
            if v.upos != "VERB":
                continue
            v_kids = kids.get(v.id, [])
            aux_pass = [c for c in v_kids if c.deprel == "aux:pass"]
            if not aux_pass:
                continue
            subj = next((c for c in v_kids if c.deprel in {"nsubj:pass", "nsubj"}), None)
            if not subj or self._is_invalid_subj(subj):
                continue
            compl = next((c for c in v_kids if c.deprel in {"obl", "obl:arg", "obj", "xcomp"}), None)
            if not compl:
                continue
            rel_ids = {v.id, *[a.id for a in aux_pass]}
            compl_kids = kids.get(compl.id, [])
            case_tok = next((c for c in compl_kids if c.deprel == "case"), None)
            if case_tok:
                rel_ids.add(case_tok.id)
            rel_raw = normalize_space(" ".join(id2tok[i].text for i in sorted(rel_ids) if i in id2tok))
            rel = expand_contractions(rel_raw)
            subj_span = self._expand_subj(subj.id, toks, kids)
            exclude = {case_tok.id} if case_tok else set()
            compl_span = self._expand_arg(compl.id, toks, kids, allow_conj=True, exclude_ids=exclude)
            arg1 = self._span_text(subj_span, toks)
            arg2 = self._span_text(compl_span, toks)
            if arg1 and arg2 and rel and not self._has_double_prep(rel, arg2):
                triplas.append({
                    "sentence": sentence, "arg1": arg1, "rel": rel, "arg2": arg2,
                    "verb_id": v.id, "subj_id": subj.id, "obj_id": compl.id,
                    "pattern_ud": "aux:pass", "subj_deprel": subj.deprel, "obj_deprel": compl.deprel,
                })
        return triplas

    def _extract_copula(self, sentence: str, toks: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[Dict[str, Any]]:
        triplas = []
        if self.config.cop_mode == "off":
            return triplas
        id2tok = {t.id: t for t in toks}
        for pred in toks:
            pred_kids = kids.get(pred.id, [])
            cop = next((c for c in pred_kids if c.deprel == "cop"), None)
            if not cop or (cop.lemma or cop.text).lower() not in COPULA_VERBS:
                continue
            subj = next((c for c in pred_kids if c.deprel == "nsubj"), None)
            if not subj or self._is_invalid_subj(subj):
                continue
            subj_span = self._expand_subj(subj.id, toks, kids)
            arg1 = self._span_text(subj_span, toks)

            simple_span = [pred.id]
            longer_span = {pred.id}
            for c in pred_kids:
                if c.deprel in {"det", "amod", "nummod", "nmod", "nmod:poss", "case", "flat", "flat:name"}:
                    longer_span.add(c.id)
                elif c.deprel == "advmod" and (c.lemma or c.text).lower() not in ADVMOD_EXCLUDE:
                    longer_span.add(c.id)
            arg2_simple = self._span_text(sorted(simple_span), toks)
            arg2_long = self._span_text(sorted(longer_span), toks)

            variants: List[Tuple[str, str]] = []
            rel_simple = expand_contractions(cop.text)
            if self.config.cop_mode == "full":
                variants.append((rel_simple, arg2_simple))
            if self.config.prefer_longer_rel_copula and arg2_long and arg2_long != arg2_simple:
                rel_long = rel_simple
                arg2_long_tokens = arg2_long.split()
                if len(arg2_long_tokens) >= 2 and arg2_long_tokens[1] in PREPS:
                    rel_long = f"{rel_simple} {arg2_long_tokens[0]} {arg2_long_tokens[1]}"
                    arg2_long2 = " ".join(arg2_long_tokens[2:]).strip()
                    if arg2_long2:
                        variants.append((rel_long, arg2_long2))
                if self.config.cop_mode == "full":
                    variants.append((rel_simple, arg2_long))
            if self.config.cop_mode == "restricted":
                restricted = []
                for rel, arg2 in variants:
                    rel_toks = rel.split()
                    arg2_toks = arg2.split()
                    if len(rel_toks) >= 3 or (len(rel_toks) >= 2 and rel_toks[-1] in PREPS and len(arg2_toks) >= 1):
                        restricted.append((rel, arg2))
                variants = restricted

            seen_local = set()
            for rel, arg2 in variants:
                key = (canon_rel(rel), canon_arg(arg2))
                if key in seen_local:
                    continue
                seen_local.add(key)
                if arg1 and arg2 and rel:
                    triplas.append({
                        "sentence": sentence, "arg1": arg1, "rel": rel, "arg2": arg2,
                        "verb_id": cop.id, "subj_id": subj.id, "obj_id": pred.id,
                        "pattern_ud": "cop", "subj_deprel": subj.deprel, "obj_deprel": pred.deprel,
                    })
        return triplas

    def _attn_pack(self, sentence: str, toks: List[UDTok]):
        if self.config.no_attn:
            return None, None, None, None
        enc = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=self.config.max_tokens_bert)
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**inputs, output_attentions=True)
        atts = list(out.attentions)
        mapping: Dict[int, List[int]] = {t.id: [] for t in toks}
        for wp_idx, (a, b) in enumerate(offsets):
            if a == 0 and b == 0:
                continue
            for t in toks:
                if not (t.end_char <= a or b <= t.start_char):
                    mapping[t.id].append(wp_idx)
        M_s = average_selected_heads(atts, self.heads_s, self.config.window_s)
        M_o = average_selected_heads(atts, self.heads_o, self.config.window_o)
        return atts, mapping, M_s, M_o

    def _attach_attention(self, row: Dict[str, Any], toks: List[UDTok], atts, mapping, M_s, M_o) -> Dict[str, Any]:
        row = dict(row)
    
        # ------------------------------------------------------------------
        # Campos sempre presentes
        # ------------------------------------------------------------------
        row["conf_att_subj"] = None
        row["conf_att_obj"] = None
        row["conf_att"] = None
    
        row["att_evidence_obj"] = []
        row["att_evidence_subj"] = []
    
        row["best_layer_obj"] = None
        row["best_head_obj"] = None
        row["best_head_score_obj"] = None
    
        row["best_layer_subj"] = None
        row["best_head_subj"] = None
        row["best_head_score_subj"] = None
    
        row["selected_heads_obj"] = [{"layer": int(l), "head": int(h)} for l, h in getattr(self, "heads_o", [])]
        row["selected_heads_subj"] = [{"layer": int(l), "head": int(h)} for l, h in getattr(self, "heads_s", [])]
    
        # ------------------------------------------------------------------
        # Se atenção estiver desligada ou indisponível, retorna a linha com
        # a estrutura mínima já preenchida
        # ------------------------------------------------------------------
        if self.config.no_attn or atts is None:
            return row
    
        v_wp = mapping.get(row.get("verb_id"), []) or []
        s_wp = mapping.get(row.get("subj_id"), []) or []
        o_wp = mapping.get(row.get("obj_id"), []) or []
    
        # Sem verbo mapeado em wordpiece não há como calcular atenção útil
        if not v_wp:
            return row
    
        conf_s = attn_score_pair(M_s, v_wp, s_wp) if s_wp else 0.0
        conf_o = attn_score_pair(M_o, v_wp, o_wp) if o_wp else 0.0
    
        row["conf_att_subj"] = float(conf_s)
        row["conf_att_obj"] = float(conf_o)
    
        # média só das partes disponíveis
        vals = []
        if s_wp:
            vals.append(float(conf_s))
        if o_wp:
            vals.append(float(conf_o))
        row["conf_att"] = float(sum(vals) / len(vals)) if vals else 0.0
    
        detailed_o = []
        if o_wp:
            for (L, H) in getattr(self, "heads_o", []):
                if not (self.config.window_o[0] <= L <= self.config.window_o[1]):
                    continue
                try:
                    M_lh = atts[L - 1][0, H - 1, :, :]
                    detailed_o.append({
                        "layer": int(L),
                        "head": int(H),
                        "score": float(attn_score_pair(M_lh, v_wp, o_wp))
                    })
                except Exception:
                    continue
    
        detailed_s = []
        if s_wp:
            for (L, H) in getattr(self, "heads_s", []):
                if not (self.config.window_s[0] <= L <= self.config.window_s[1]):
                    continue
                try:
                    M_lh = atts[L - 1][0, H - 1, :, :]
                    detailed_s.append({
                        "layer": int(L),
                        "head": int(H),
                        "score": float(attn_score_pair(M_lh, v_wp, s_wp))
                    })
                except Exception:
                    continue
    
        detailed_o.sort(key=lambda x: x["score"], reverse=True)
        detailed_s.sort(key=lambda x: x["score"], reverse=True)
    
        row["att_evidence_obj"] = detailed_o[:10]
        row["att_evidence_subj"] = detailed_s[:10]
    
        best_o = detailed_o[0] if detailed_o else None
        best_s = detailed_s[0] if detailed_s else None
    
        if best_o is not None:
            row["best_layer_obj"] = int(best_o["layer"])
            row["best_head_obj"] = int(best_o["head"])
            row["best_head_score_obj"] = float(best_o["score"])
    
        if best_s is not None:
            row["best_layer_subj"] = int(best_s["layer"])
            row["best_head_subj"] = int(best_s["head"])
            row["best_head_score_subj"] = float(best_s["score"])

        row["attn_decision_pattern"] = self._pattern_uses_attention_decision(row.get("pattern_ud", "unknown"))
        row["attn_threshold"] = None if self.config.no_attn else self.config.attn_threshold

        if self.config.no_attn or atts is None:
            row["attn_decision_pattern"] = False
            row["attn_threshold"] = None if self.config.no_attn else self.config.attn_threshold
            return row
    
        return row

    def _pattern_uses_attention_decision(self, pattern_ud: str) -> bool:
        if self.config.no_attn:
            return False
        if not getattr(self.config, "attn_decision_enabled", True):
            return False
        allowed = set(getattr(self.config, "attn_decision_patterns", ()))
        return pattern_ud in allowed
    
    
    def _passes_attention_gate(self, row: Dict[str, Any]) -> bool:
        """
        Aplica threshold de atenção somente em padrões selecionados.
        """
        if self.config.no_attn:
            return True
    
        pattern = row.get("pattern_ud", "unknown")
        if not self._pattern_uses_attention_decision(pattern):
            return True
    
        conf = row.get("conf_att", None)
    
        if conf is None:
            return bool(getattr(self.config, "attn_keep_if_missing", True))
    
        try:
            return float(conf) >= float(self.config.attn_threshold)
        except Exception:
            return bool(getattr(self.config, "attn_keep_if_missing", True))

    def extract(self, sentence: str) -> Tuple[List[Dict[str, Any]], List[UDTok]]:
        ud = self._parse_ud(sentence)
        kids = self._children_index(ud)
        all_rows: List[Dict[str, Any]] = []
        best_per_verb: Dict[int, Dict[str, Any]] = {}

        for v in ud:
            if v.upos not in {"VERB", "AUX"} or v.deprel in CLAUSE_SKIP:
                continue
            v_kids = kids.get(v.id, [])
            subj = next((c for c in v_kids if c.deprel in {"nsubj", "nsubj:pass"}), None)
            obj = next((c for c in v_kids if c.deprel == "obj" or c.deprel.startswith("obj:")), None)
            if subj is None or obj is None or self._is_invalid_subj(subj):
                continue
            rel, _, prep_excl = self._build_rel(v, v_kids, obj.id, ud, kids)
            subj_span = self._expand_subj(subj.id, ud, kids)
            if self.config.extract_coord:
                coord_objs = self._extract_coordinated_elements(obj, kids, ud)
                if coord_objs and len(coord_objs) > 1:
                    for obj_span in coord_objs:
                        arg1 = self._span_text(subj_span, ud)
                        arg2 = self._span_text([i for i in obj_span if i not in prep_excl], ud)
                        if arg1 and arg2 and rel and not self._has_double_prep(rel, arg2):
                            all_rows.append({
                                "sentence": sentence, "arg1": arg1, "rel": rel, "arg2": arg2,
                                "verb_id": v.id, "subj_id": subj.id, "obj_id": obj.id,
                                "pattern_ud": "verb+obj[coord]", "subj_deprel": subj.deprel, "obj_deprel": obj.deprel,
                            })
                    continue
            obj_span = self._expand_obj(obj.id, ud, kids, allow_conj=True, exclude_ids=prep_excl)
            if self._is_clitic_obj(obj, obj_span):
                continue
            arg1 = self._span_text(subj_span, ud)
            arg2 = self._span_text(obj_span, ud)
            if self._has_double_prep(rel, arg2):
                continue

            def make_row(a1, rl, a2, tag="verb+obj"):
                return {
                    "sentence": sentence, "arg1": a1, "rel": rl, "arg2": a2,
                    "verb_id": v.id, "subj_id": subj.id, "obj_id": obj.id,
                    "pattern_ud": tag, "subj_deprel": subj.deprel, "obj_deprel": obj.deprel,
                }

            cands = [make_row(arg1, rel, arg2)]
            if self.config.generate_obl_variants:
                for obl_tok in [c for c in v_kids if c.deprel in self.config.obl_deps and c.id > obj.id]:
                    obl_sp = self._expand_arg(obl_tok.id, ud, kids, allow_appos=True, allow_conj=True)
                    combined = sorted(set(obj_span) | set(obl_sp))
                    comb_text = self._span_text(combined, ud)
                    if comb_text != arg2 and not self._has_double_prep(rel, comb_text):
                        cands.append(make_row(arg1, rel, comb_text, "verb+obj[+obl]"))
                    break
            if self.config.deduplicate_by_verb:
                base = cands[0]
                if v.id not in best_per_verb:
                    best_per_verb[v.id] = base
                all_rows.extend(cands[1:])
            else:
                all_rows.extend(cands)

        if self.config.deduplicate_by_verb:
            all_rows = list(best_per_verb.values()) + all_rows
        if self.config.extract_aux_pass:
            all_rows.extend(self._extract_aux_pass(sentence, ud, kids))
        if self.config.extract_cop:
            all_rows.extend(self._extract_copula(sentence, ud, kids))
        if self.config.extract_conj_verbs:
            all_rows.extend(self._extract_coordinated_verbs(sentence, ud, kids))
        if self.config.extract_modal:
            all_rows.extend(self._extract_modal(sentence, ud, kids))

        atts, mapping, M_s, M_o = self._attn_pack(sentence, ud)

        enriched = []
        seen: Set[Tuple[str, str, str]] = set()
        
        for r in all_rows:
            r = self._attach_attention(r, ud, atts, mapping, M_s, M_o)

            # não filtra aqui; apenas anexa atenção
            if getattr(self.config, "attn_filter_in_legacy", False):
                if not self.config.no_attn and r["conf_att"] < self.config.attn_threshold:
                    continue
        
            key = (canon_arg(r["arg1"]), canon_rel(r["rel"]), canon_arg(r["arg2"]))
            if key in seen:
                continue
        
            seen.add(key)
            enriched.append(r)
        
        return enriched, ud




def _find_span_token_ids(sentence: str, toks: List[UDTok], span_text: str) -> List[int]:
    span_text = normalize_space(span_text or "")
    if not span_text:
        return []
    low_sent = sentence.lower()
    low_span = span_text.lower()
    start = low_sent.find(low_span)
    if start >= 0:
        end = start + len(low_span)
        return [t.id for t in toks if t.start_char < end and t.end_char > start]
    span_words = set(canon_arg(span_text).split())
    return [t.id for t in toks if canon_arg(t.text) in span_words]


def infer_surface_pattern_from_gold(extractor: OpenIEExtractorBIA, sentence: str, g: Dict[str, Any]) -> Dict[str, Any]:
    toks = extractor._parse_ud(sentence)
    kids = extractor._children_index(toks)
    subj_ids = set(_find_span_token_ids(sentence, toks, g.get("arg1", "")))
    obj_ids = set(_find_span_token_ids(sentence, toks, g.get("arg2", "")))
    rel_norm = canon_rel(g.get("rel", ""))
    subj_deprel = None
    obj_deprel = None
    if subj_ids:
        subj_deprel = next((t.deprel for t in toks if t.id in subj_ids and t.deprel.startswith("nsubj")), None)
    if obj_ids:
        obj_deprel = next((t.deprel for t in toks if t.id in obj_ids and t.deprel in {"obj", "iobj", "obl", "obl:arg", "xcomp", "nmod"}), None)
    pattern = "unknown"
    rel_has_cop = any((t.lemma or t.text).lower() in COPULA_VERBS for t in toks if canon_rel(t.text) in rel_norm.split()) or rel_norm.split()[:1] in (["é"],["foi"],["era"],["são"],["está"],["estão"])
    # direct scan by verbal heads
    for v in toks:
        if v.upos not in {"VERB", "AUX", "ADJ", "NOUN"}:
            continue
        deps = {c.deprel for c in kids.get(v.id, [])}
        if rel_has_cop and "cop" in deps and (subj_ids and (v.id in obj_ids or any(c.id in subj_ids for c in kids.get(v.id, []) if c.deprel.startswith("nsubj")))):
            pattern = "cop"
            obj_deprel = obj_deprel or v.deprel
            break
        if "nsubj" in deps or "nsubj:pass" in deps:
            if "obj" in deps or any(d.startswith("obj:") for d in deps):
                pattern = "verb+obj[+obl]" if ("obl" in deps or "obl:arg" in deps) else "verb+obj"
                break
            if "obl" in deps or "obl:arg" in deps:
                pattern = "verb+obl"
                break
            if any(c.deprel == "conj" and c.upos == "VERB" for c in kids.get(v.id, [])):
                pattern = "verb+obj[coord]"
    return {
        "gold_pattern_ud_surface": pattern,
        "gold_subj_deprel_surface": subj_deprel,
        "gold_obj_deprel_surface": obj_deprel,
    }

# -----------------------------------------------------------------------------
# Matching / avaliação
# -----------------------------------------------------------------------------
def args_match(pred_arg: str, gold_arg: str, config: Config) -> bool:
    pp, gg = canon_arg(pred_arg), canon_arg(gold_arg)
    if pp == gg:
        return True
    lo, sh = (pp, gg) if len(pp) >= len(gg) else (gg, pp)
    if sh and lo.endswith(sh):
        return True
    if config.use_prefix_match:
        if len(pp) <= len(gg) and gg.startswith(pp) and len(pp.split()) >= config.prefix_min_tokens:
            return True
        if len(gg) <= len(pp) and pp.startswith(gg) and len(gg.split()) >= config.prefix_min_tokens:
            return True
    if config.use_substring_match:
        pp_words = set(pp.split())
        gg_words = set(gg.split())
        if pp_words and gg_words and (pp_words <= gg_words or gg_words <= pp_words):
            return True
    return False


def rels_match(pred_rel: str, gold_rel: str, config: Config) -> bool:
    pr, gr = canon_rel(pred_rel), canon_rel(gold_rel)
    if pr == gr:
        return True
    if config.use_partial_rel:
        pw, gw = pr.split(), gr.split()
        if pw and gw and (pw == gw[: len(pw)] or gw == pw[: len(gw)]):
            return True
        if gw and gw[0] in GOLD_REL_SKIP:
            gw2 = gw[1:]
            if gw2 and (pw == gw2[: len(pw)] or gw2 == pw[: len(gw2)]):
                return True
    return False


def triple_matches(pred: Dict[str, Any], gold: Dict[str, Any], config: Config) -> bool:
    return rels_match(pred["rel"], gold["rel"], config) and args_match(pred["arg1"], gold["arg1"], config) and args_match(pred["arg2"], gold["arg2"], config)


def evaluate_dataset(preds_by_sent: List[List[Dict[str, Any]]], gold_rows: List[Dict[str, Any]], config: Config) -> Dict[str, Any]:
    tp = fp = fn = 0
    details = []
    fp_by_rel = Counter()
    fn_by_pattern = Counter()
    tp_by_pattern = Counter()
    per_pattern = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for item, preds in zip(gold_rows, preds_by_sent):
        golds = [g for g in item["gold"] if g.get("valid", True)]
        matched_pred = [False] * len(preds)
        matched_gold = [False] * len(golds)
        sentence = item["sentence"]
        local = {
            "sentence": sentence,
            "pred": preds,
            "gold": golds,
            "pred_matched": matched_pred,
            "gold_matched": matched_gold,
        }
        for gi, g in enumerate(golds):
            for pi, p in enumerate(preds):
                if matched_pred[pi]:
                    continue
                if triple_matches(p, g, config):
                    matched_pred[pi] = True
                    matched_gold[gi] = True
                    pat = p.get("pattern_ud") or "unknown"
                    tp += 1
                    tp_by_pattern[pat] += 1
                    per_pattern[pat]["TP"] += 1
                    break
        for pi, p in enumerate(preds):
            if not matched_pred[pi]:
                fp += 1
                fp_by_rel[p.get("rel", "")] += 1
                pat = p.get("pattern_ud") or "unknown"
                per_pattern[pat]["FP"] += 1
        for gi, g in enumerate(golds):
            if not matched_gold[gi]:
                fn += 1
                pat = g.get("gold_pattern_ud") or "unknown"
                fn_by_pattern[pat] += 1
                per_pattern[pat]["FN"] += 1
        details.append(local)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = jaccard_accuracy(tp, fp, fn)

    per_pattern_rows = []
    for pat, c in per_pattern.items():
        p = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0
        r = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_pattern_rows.append({"pattern_ud": pat, "tp": c["TP"], "fp": c["FP"], "fn": c["FN"], "precision": p, "recall": r, "f1": f})
    per_pattern_rows.sort(key=lambda x: x["f1"], reverse=True)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "details": details,
        "fp_by_rel": fp_by_rel,
        "fn_by_pattern": fn_by_pattern,
        "tp_by_pattern": tp_by_pattern,
        "per_pattern": per_pattern_rows,
    }


# -----------------------------------------------------------------------------
# Enriquecimento gold
# -----------------------------------------------------------------------------
def infer_gold_patterns(extractor: OpenIEExtractorBIA, gold_rows: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    enriched_rows = []
    for item in tqdm(gold_rows, desc="Inferindo padrões UD do gold"):
        sentence = item["sentence"]
        candidates, _ = extractor.extract(sentence)
        golds = []
        for g in item["gold"]:
            g2 = dict(g)
            g2["gold_pattern_ud_matched"] = "unknown"
            g2["gold_subj_deprel_matched"] = None
            g2["gold_obj_deprel_matched"] = None
            g2["gold_match_conf_att"] = None
            g2.update(infer_surface_pattern_from_gold(extractor, sentence, g2))
            best = None
            for cand in candidates:
                if triple_matches(cand, g2, config):
                    best = cand
                    break
            if best is not None:
                g2["gold_pattern_ud_matched"] = best.get("pattern_ud", "unknown")
                g2["gold_subj_deprel_matched"] = best.get("subj_deprel")
                g2["gold_obj_deprel_matched"] = best.get("obj_deprel")
                g2["gold_match_conf_att"] = best.get("conf_att")
            golds.append(g2)
        row = dict(item)
        row["gold"] = golds
        enriched_rows.append(row)
    return enriched_rows


# -----------------------------------------------------------------------------
# Pipelines de saída
# -----------------------------------------------------------------------------
def flatten_gold_rows(gold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat = []
    for item in gold_rows:
        sent = item["sentence"]
        for g in item["gold"]:
            flat.append({
                "sentence": sent,
                "arg1": g.get("arg1", ""),
                "rel": g.get("rel", ""),
                "arg2": g.get("arg2", ""),
                "gold_pattern_ud_surface": g.get("gold_pattern_ud_surface", "unknown"),
                "gold_subj_deprel_surface": g.get("gold_subj_deprel_surface"),
                "gold_obj_deprel_surface": g.get("gold_obj_deprel_surface"),
                "gold_pattern_ud_matched": g.get("gold_pattern_ud_matched", "unknown"),
                "gold_subj_deprel_matched": g.get("gold_subj_deprel_matched"),
                "gold_obj_deprel_matched": g.get("gold_obj_deprel_matched"),
                "gold_match_conf_att": g.get("gold_match_conf_att"),
                "valid": g.get("valid", True),
            })
    return flat


def flatten_pred_rows(preds_by_sent: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for preds in preds_by_sent:
        for p in preds:
            q = dict(p)
            q["att_evidence_obj"] = json.dumps(q.get("att_evidence_obj", []), ensure_ascii=False)
            q["att_evidence_subj"] = json.dumps(q.get("att_evidence_subj", []), ensure_ascii=False)
            q["selected_heads_obj"] = json.dumps(q.get("selected_heads_obj", []), ensure_ascii=False)
            q["selected_heads_subj"] = json.dumps(q.get("selected_heads_subj", []), ensure_ascii=False)
            rows.append(q)
    return rows


def build_triples_table(preds_by_sent: List[List[Dict[str, Any]]], gold_rows: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    rows = []
    for item, preds in zip(gold_rows, preds_by_sent):
        sent = item["sentence"]
        golds = [g for g in item["gold"] if g.get("valid", True)]
        norm_gold = {(canon_arg(g["arg1"]), canon_rel(g["rel"]), canon_arg(g["arg2"])) for g in golds}
        norm_pred = {(canon_arg(p["arg1"]), canon_rel(p["rel"]), canon_arg(p["arg2"])) for p in preds}
        for p in preds:
            key = (canon_arg(p["arg1"]), canon_rel(p["rel"]), canon_arg(p["arg2"]))
            rows.append({
                "sentence": sent,
                "arg1": p["arg1"], "rel": p["rel"], "arg2": p["arg2"],
                "gold": 1 if key in norm_gold else 0,
                "pred": 1,
                "score": p.get("conf_att", 1.0),
                "pattern_ud": p.get("pattern_ud"),
                "subj_deprel": p.get("subj_deprel"),
                "obj_deprel": p.get("obj_deprel"),
            })
        for g in golds:
            key = (canon_arg(g["arg1"]), canon_rel(g["rel"]), canon_arg(g["arg2"]))
            if key not in norm_pred:
                rows.append({
                    "sentence": sent,
                    "arg1": g["arg1"], "rel": g["rel"], "arg2": g["arg2"],
                    "gold": 1,
                    "pred": 0,
                    "score": 0.0,
                    "pattern_ud": g.get("gold_pattern_ud_surface", "unknown"),
                    "subj_deprel": g.get("gold_subj_deprel"),
                    "obj_deprel": g.get("gold_obj_deprel"),
                })
    return rows



def aggregate_head_usage(preds_by_sent: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    from collections import Counter, defaultdict

    overall = Counter()
    by_pattern = defaultdict(Counter)

    for sent_preds in preds_by_sent:
        for row in sent_preds:
            pattern = row.get("pattern_ud", "unknown")

            # prioridade 1: selected_heads explícitos
            subj_heads = row.get("selected_heads_subj")
            obj_heads = row.get("selected_heads_obj")

            used_any = False

            def register_head(head_item, role: str):
                nonlocal used_any
                if isinstance(head_item, dict):
                    layer = head_item.get("layer")
                    head = head_item.get("head")
                    if layer is None or head is None:
                        return
                    key = f"L{layer}_H{head}"
                elif isinstance(head_item, (list, tuple)) and len(head_item) >= 2:
                    key = f"L{head_item[0]}_H{head_item[1]}"
                else:
                    return

                overall[key] += 1
                by_pattern[pattern][key] += 1
                used_any = True

            if isinstance(subj_heads, list):
                for h in subj_heads:
                    register_head(h, "subj")
            if isinstance(obj_heads, list):
                for h in obj_heads:
                    register_head(h, "obj")

            # fallback: best head subj/obj
            if not used_any:
                bls = row.get("best_layer_subj")
                bhs = row.get("best_head_subj")
                blo = row.get("best_layer_obj")
                bho = row.get("best_head_obj")

                if pd.notna(bls) and pd.notna(bhs):
                    key = f"L{int(bls)}_H{int(bhs)}"
                    overall[key] += 1
                    by_pattern[pattern][key] += 1
                    used_any = True

                if pd.notna(blo) and pd.notna(bho):
                    key = f"L{int(blo)}_H{int(bho)}"
                    overall[key] += 1
                    by_pattern[pattern][key] += 1
                    used_any = True

    overall_rows = [
        {"head": k, "count": v}
        for k, v in overall.most_common()
    ]

    by_pattern_rows = []
    for pattern, counter in by_pattern.items():
        for k, v in counter.most_common():
            by_pattern_rows.append({
                "pattern_ud": pattern,
                "head": k,
                "count": v
            })

    return {
        "overall": overall_rows,
        "by_pattern": by_pattern_rows,
        "summary": {
            "n_unique_heads_overall": len(overall),
            "n_patterns": len(by_pattern),
            "total_head_events": sum(overall.values()),
        }
    }


# -----------------------------------------------------------------------------
# CLI principal
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenIE-PT v10 BIA experiment")
    p.add_argument("--gold", required=True, help="JSONL gold (BIA/STEFANY)")
    p.add_argument("--output-dir", default="./resultados_bia_aligned", help="Diretório de saída")
    p.add_argument("--bert-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--bosque", default=None, help="Arquivo Bosque .conllu para ranking de heads")
    p.add_argument("--heads-mode", default="rank", choices=["rank", "all", "forced", "random"])
    p.add_argument("--heads-json", default=None, help="JSON com heads_s e heads_o")
    p.add_argument("--top-k-heads", type=int, default=10)
    p.add_argument("--n-sent-rank", type=int, default=300)
    p.add_argument("--window-s", default="2:5")
    p.add_argument("--window-o", default="7:12")
    p.add_argument("--attn-threshold", type=float, default=0.0)
    p.add_argument("--no-attn", action="store_true")
    p.add_argument("--eval-mode", default="strict", choices=["strict", "tolerant"])
    p.add_argument("--use-gpu-stanza", action="store_true")
    p.add_argument("--dataset-name", default="bia")
    p.add_argument("--allow-fallback-all-heads", action="store_true", help="Permite fallback para todas as heads quando rank não puder ser executado")
    p.add_argument("--cop-mode", default="full", choices=["full", "off", "restricted"], help="Controle da extração copular")
    p.add_argument("--no-coord", action="store_true", help="Desliga extração por coordenação")
    p.add_argument("--no-obl-fallback", action="store_true", help="Desliga variantes/baseadas em OBL")
    p.add_argument("--random-heads-seed", type=int, default=13, help="Seed para heads aleatórias quando heads-mode=random")
    return p


def load_forced_heads_from_json(path: Optional[str]) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    hs = obj.get("heads_s") or obj.get("subj") or obj.get("s")
    ho = obj.get("heads_o") or obj.get("obj") or obj.get("o")
    return hs, ho


def main():
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forced_s, forced_o = load_forced_heads_from_json(args.heads_json)
    config = Config(
        bert_model=args.bert_model,
        use_gpu_stanza=args.use_gpu_stanza,
        attn_threshold=args.attn_threshold,
        heads_mode=args.heads_mode,
        top_k_heads=args.top_k_heads,
        n_sent_rank=args.n_sent_rank,
        window_s=parse_window(args.window_s, (2, 5)),
        window_o=parse_window(args.window_o, (7, 12)),
        bosque_path=args.bosque,
        forced_heads_s=forced_s,
        forced_heads_o=forced_o,
        no_attn=args.no_attn,
        allow_fallback_all_heads=args.allow_fallback_all_heads,
        extract_coord=not args.no_coord,
        generate_obl_variants=not args.no_obl_fallback,
        cop_mode=args.cop_mode,
        random_heads_seed=args.random_heads_seed,
    )

    extractor = OpenIEExtractorBIA(config=config, verbose=True)
    gold_rows = read_gold_jsonl(args.gold)
    gold_rows = infer_gold_patterns(extractor, gold_rows, config)

    preds_by_sent: List[List[Dict[str, Any]]] = []
    for item in tqdm(gold_rows, desc=f"Extraindo triplas/{args.dataset_name}"):
        preds, _ = extractor.extract(item["sentence"])
        preds_by_sent.append(preds)

    metrics = evaluate_dataset(preds_by_sent, gold_rows, config)

    # dumps
    pred_jsonl_rows = []
    for item, preds in zip(gold_rows, preds_by_sent):
        pred_jsonl_rows.append({"sentence": item["sentence"], "pred": preds})
    write_jsonl(output_dir / f"{args.dataset_name}_predictions_enriched.jsonl", pred_jsonl_rows)
    write_csv(output_dir / f"{args.dataset_name}_predictions_enriched.csv", flatten_pred_rows(preds_by_sent))
    write_csv(output_dir / f"{args.dataset_name}_gold_enriched.csv", flatten_gold_rows(gold_rows))
    write_csv(output_dir / f"{args.dataset_name}_triples_table.csv", build_triples_table(preds_by_sent, gold_rows, config))
    write_csv(output_dir / f"{args.dataset_name}_per_pattern_metrics.csv", metrics["per_pattern"])

    head_usage = aggregate_head_usage(preds_by_sent)
    write_csv(output_dir / f"{args.dataset_name}_head_usage_overall.csv", head_usage["overall"])
    write_csv(output_dir / f"{args.dataset_name}_head_usage_by_pattern.csv", head_usage["by_pattern"])
    (output_dir / f"{args.dataset_name}_head_usage_summary.json").write_text(json.dumps(head_usage, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_heads = dict(extractor.heads_meta)
    selected_heads["bert_model"] = args.bert_model
    selected_heads["device"] = str(extractor.device)
    selected_heads["window_s"] = list(config.window_s)
    selected_heads["window_o"] = list(config.window_o)
    selected_heads["attn_threshold"] = config.attn_threshold
    selected_heads["allow_fallback_all_heads"] = config.allow_fallback_all_heads
    (output_dir / f"{args.dataset_name}_selected_heads.json").write_text(json.dumps(selected_heads, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset": args.dataset_name,
        "gold_path": args.gold,
        "bert_model": args.bert_model,
        "device": str(extractor.device),
        "heads_mode": config.heads_mode,
        "selected_heads_mode": extractor.heads_meta.get("mode"),
        "head_analysis_valid": extractor.heads_meta.get("head_analysis_valid", False),
        "window_s": list(config.window_s),
        "window_o": list(config.window_o),
        "attn_threshold": config.attn_threshold,
        "TP": metrics["TP"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "gold_patterns_top": metrics["fn_by_pattern"].most_common(20),
        "fp_rel_top": metrics["fp_by_rel"].most_common(20),
    }
    (output_dir / f"{args.dataset_name}_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(f"Dataset:   {args.dataset_name}")
    print(f"Device:    {extractor.device}")
    print(f"Heads req: {config.heads_mode}")
    print(f"Heads sel: {extractor.heads_meta.get("mode")}")
    print(f"Head analysis valid: {extractor.heads_meta.get("head_analysis_valid", False)}")
    print(f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Accuracy*: {metrics['accuracy']:.4f}")
    print("* Accuracy aqui = TP / (TP + FP + FN), pois OpenIE por matching de conjuntos não possui TN natural.")
    print("\nTop FN por pattern_ud:")
    for pat, n in metrics["fn_by_pattern"].most_common(10):
        print(f"  {pat:<25} {n:>4}")
    print("\nTop FP por REL:")
    for rel, n in metrics["fp_by_rel"].most_common(10):
        print(f"  {rel:<25} {n:>4}")
    print("\nArquivos gerados:")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Aliases do núcleo legado embutido
# -----------------------------------------------------------------------------
LegacyConfig = Config
LegacyOpenIEExtractorBIA = OpenIEExtractorBIA
evaluate_dataset_legacy = evaluate_dataset
infer_surface_pattern_from_gold_legacy = infer_surface_pattern_from_gold
flatten_pred_rows_legacy = flatten_pred_rows
flatten_gold_rows_legacy = flatten_gold_rows
build_triples_table_legacy = build_triples_table

# -----------------------------------------------------------------------------
# Configuração teórica
# -----------------------------------------------------------------------------
@dataclass
class Config(LegacyConfig):
    apply_theoretical_rules: bool = True
    theory_mode: str = "filter"  # off | annotate | filter
    strict_theory: bool = False
    allow_rule_warnings: bool = True

    apply_E1: bool = True
    apply_E2: bool = True
    apply_E3: bool = True
    apply_E4: bool = True
    apply_E4_1: bool = True
    apply_E4_2: bool = True
    apply_E5: bool = True
    apply_E6: bool = True
    apply_E7: bool = True
    apply_E8: bool = True
    apply_E9: bool = True
    apply_S1: bool = True
    apply_S2: bool = True
    apply_S3: bool = True
    apply_S4: bool = True
    apply_S5: bool = True

    s2_span_policy: str = "both"  # minimal | extensive | both
    s1_keep_negation: bool = True
    s1_block_conditionals: bool = True
    s1_block_reported_belief: bool = False
    s3_decompose_nested: bool = True
    e8_allow_clausal_args: bool = True

    prepositional_locutions: Set[str] = field(default_factory=lambda: {
        "ao longo de", "de acordo com", "por meio de", "em frente a",
        "em relação a", "com base em", "a partir de", "devido a",
    })
    reporting_verbs: Set[str] = field(default_factory=lambda: {
        "dizer", "afirmar", "alegar", "informar", "declarar", "relatar",
    })
    belief_verbs: Set[str] = field(default_factory=lambda: {
        "achar", "acreditar", "pensar", "supor", "imaginar", "considerar",
    })
    conditional_markers: Set[str] = field(default_factory=lambda: {
        "se", "caso", "desde que", "contanto que", "a menos que",
    })
    mwe_relational_starts: Set[str] = field(default_factory=lambda: {
        "ao", "de", "por", "em", "a",
    })

from dataclasses import dataclass, field

@dataclass
class CandidateTriple:
    sentence: str
    arg1: str
    rel: str
    arg2: str
    verb_id: Optional[int] = None
    subj_id: Optional[int] = None
    obj_id: Optional[int] = None
    pattern_ud: str = "unknown"
    subj_deprel: Optional[str] = None
    obj_deprel: Optional[str] = None
    source_rule: str = "legacy"
    candidate_rank: int = 0
    variant_type: str = "base"
    modal_scope: Optional[str] = None
    voice: Optional[str] = None

    conf_att_subj: Optional[float] = None
    conf_att_obj: Optional[float] = None
    conf_att: Optional[float] = None

    best_layer_subj: Optional[int] = None
    best_head_subj: Optional[int] = None
    best_layer_obj: Optional[int] = None
    best_head_obj: Optional[int] = None

    att_evidence_subj: List[Any] = field(default_factory=list)
    att_evidence_obj: List[Any] = field(default_factory=list)

    selected_heads_subj: List[Any] = field(default_factory=list)
    selected_heads_obj: List[Any] = field(default_factory=list)

    heuristics_used: List[str] = field(default_factory=list)
    theory_rules_passed: List[str] = field(default_factory=list)
    theory_rules_failed: List[str] = field(default_factory=list)


@dataclass
class TheoryValidation:
    rule_E1: Optional[bool] = None
    rule_E2: Optional[bool] = None
    rule_E3: Optional[bool] = None
    rule_E4: Optional[bool] = None
    rule_E4_1: Optional[bool] = None
    rule_E4_2: Optional[bool] = None
    rule_E5: Optional[bool] = None
    rule_E6: Optional[bool] = None
    rule_E7: Optional[bool] = None
    rule_E8: Optional[bool] = None
    rule_E9: Optional[bool] = None
    rule_S1: Optional[bool] = None
    rule_S2: Optional[bool] = None
    rule_S3: Optional[bool] = None
    rule_S4: Optional[bool] = None
    rule_S5: Optional[bool] = None
    theory_valid: bool = True
    theory_blocked: bool = False
    theory_score: float = 1.0
    critical_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    factuality_label: str = "asserted"
    arg1_head_type: str = "OTHER"
    arg2_head_type: str = "OTHER"


# -----------------------------------------------------------------------------
# Extrator refatorado
# -----------------------------------------------------------------------------
class OpenIEExtractorBIA(LegacyOpenIEExtractorBIA):
    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.config = config or Config()
        super().__init__(config=self.config, verbose=verbose)

    # -------------------------
    # Utilidades teóricas
    # -------------------------
    def _expand_arg_minimal(self, tok_id: int, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[int]:
        keep = {"det", "amod", "nummod", "compound", "flat", "flat:name"}
        ids = {tok_id}
        for ch in kids.get(tok_id, []):
            if ch.deprel in keep:
                ids.add(ch.id)
        return sorted(ids)

    def _expand_arg_extensive(self, tok_id: int, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[int]:
        return self._expand_arg(tok_id, ud, kids, allow_appos=not self.config.disable_appos_alias, allow_conj=True)

    def _prune_nonessential_material(self, span_ids: List[int], ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[int]:
        """
        Remove material não essencial de spans argumentais:
        - relativas e parataxe
        - conteúdo entre parênteses
        - metalinguagem e glosas
        """
        if not span_ids:
            return span_ids
    
        tok_by_id = {t.id: t for t in ud}
        ordered = [tok_by_id[i] for i in sorted(span_ids) if i in tok_by_id]
    
        pruned_ids = []
        paren_depth = 0
    
        bad_words = {
            "latim", "inglês", "português", "francês", "espanhol",
            "abreviado", "abreviada", "abreviados", "abreviaturas",
            "sigla", "siglas",
        }
    
        for tok in ordered:
            txt = canon_arg(tok.text or "")
            dep = (tok.deprel or "").lower()
    
            if tok.text == "(":
                paren_depth += 1
                continue
            if tok.text == ")":
                paren_depth = max(0, paren_depth - 1)
                continue
            if paren_depth > 0:
                continue
    
            if dep in {"acl", "acl:relcl", "parataxis", "discourse"}:
                continue
    
            if txt in {",", ";", ":"}:
                continue
    
            if txt in bad_words:
                continue
    
            pruned_ids.append(tok.id)
    
        return sorted(set(pruned_ids))

    def _truncate_argument_span(self, span_ids: List[int], ud: List[UDTok]) -> List[int]:
        """
        Trunca o span argumental ao detectar início de glosa/parentético
        ou marcador metalinguístico.
        """
        if not span_ids:
            return span_ids
    
        tok_by_id = {t.id: t for t in ud}
        ordered = [tok_by_id[i] for i in sorted(span_ids) if i in tok_by_id]
    
        keep = []
        for i, tok in enumerate(ordered):
            txt = canon_arg(tok.text or "")
    
            # corta ao entrar em parentético
            if tok.text == "(":
                break
    
            # corta ao chegar em glosa metalinguística
            if txt in {"latim", "inglês", "português", "francês", "espanhol"}:
                break
    
            # corta em vírgula forte, típico de aposto/glosa
            if tok.text == "," and i > 0:
                break
    
            keep.append(tok.id)
    
        return keep

    def _build_rel_minimal(self, v: UDTok, v_kids: List[UDTok], obj_id: Optional[int], ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> Tuple[str, List[int], Set[int]]:
        return self._build_rel(v, v_kids, obj_id, ud, kids)

    def _attach_governed_preposition_to_rel(self, rel: str, obj: Optional[UDTok], kids: Dict[int, List[UDTok]]) -> str:
        if obj is None:
            return rel
        case_children = [c.text for c in kids.get(obj.id, []) if c.deprel == "case"]
        if not case_children:
            return rel
        prep = " ".join(case_children)
        if canon_rel(prep) and canon_rel(prep) not in canon_rel(rel):
            return normalize_space(f"{rel} {prep}")
        return rel

    def _detect_prepositional_locution(self, rel: str) -> Optional[str]:
        rel_c = canon_rel(rel)
        for loc in self.config.prepositional_locutions:
            if canon_rel(loc) in rel_c:
                return loc
        return None

    def _expand_rel_with_prepositional_locution(self, rel: str) -> str:
        return rel

    def _decompose_contracted_rel_tokens(self, rel: str, arg2: str) -> Tuple[str, str]:
        rel = expand_contractions(rel)
        arg2 = expand_contractions(arg2)
        return normalize_space(rel), normalize_space(arg2)

    def _classify_argument_head_type(self, arg_text: str, ud: List[UDTok]) -> str:
        toks = [t for t in ud if canon_arg(t.text) and canon_arg(t.text) in canon_arg(arg_text)]
        priority = ["PRON", "PROPN", "NOUN", "VERB", "ADJ"]
        for p in priority:
            if any((t.upos or "").upper() == p for t in toks):
                return p
        if any((t.deprel or "").startswith(("ccomp", "xcomp", "csubj")) for t in toks):
            return "CLAUSE"
        return "OTHER"

    def _detect_nonessential_relative_clause(self, arg_text: str, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        arg_c = canon_arg(arg_text)
        relcl_tokens = [t for t in ud if t.deprel == "acl:relcl" and canon_arg(t.text) in arg_c]
        has_commas = "," in arg_text
        return bool(relcl_tokens and has_commas)

    def _detect_conditional_scope_for_verb(
        self,
        verb_id: Optional[int],
        ud: List[UDTok],
        kids: Dict[int, List[UDTok]]
    ) -> bool:
        """
        Detecta condicional apenas quando houver marcador subordinativo real
        no escopo local do verbo. Não confunde 'se' clítico/reflexivo com
        'se' condicional.
        """
        if verb_id is None:
            return False
    
        verb = next((t for t in ud if t.id == verb_id), None)
        if verb is None:
            return False
    
        local_ids = {verb.id}
        for ch in kids.get(verb.id, []):
            local_ids.add(ch.id)
            for gch in kids.get(ch.id, []):
                local_ids.add(gch.id)
    
        local_toks = [t for t in ud if t.id in local_ids]
    
        conditional_multi = {
            canon_arg(x)
            for x in getattr(self.config, "conditional_markers", [])
            if " " in x
        }
        local_text = canon_arg(" ".join(t.text for t in local_toks))
        for marker in conditional_multi:
            if re.search(rf"\b{re.escape(marker)}\b", local_text):
                return True
    
        for t in local_toks:
            txt = canon_arg(t.text or "")
            lem = canon_arg(t.lemma or "")
    
            # Só conta se for realmente subordinador/conjunção
            is_conditional_token = (
                txt == "se" or lem == "se" or txt in {"caso", "contanto"} or lem in {"caso", "contanto"}
            )
            if not is_conditional_token:
                continue
    
            upos = (t.upos or "").upper()
            dep = (t.deprel or "").lower()
    
            # 'se' como PRON/obj/expl/etc. NÃO é condicional
            if upos == "PRON":
                continue
    
            # Casos aceitáveis como condicional
            if upos in {"SCONJ", "CCONJ", "ADV"}:
                return True
            if dep.startswith("mark") or dep in {"advmod", "discourse"}:
                return True
    
        return False

    def _detect_reported_belief_or_speech(self, rel: str) -> Tuple[bool, Optional[str]]:
        rr = canon_rel(rel)
        for v in sorted(self.config.reporting_verbs | self.config.belief_verbs):
            if canon_rel(v) in rr.split() or re.search(rf"\b{re.escape(canon_rel(v))}\b", rr):
                return True, v
        return False, None

    def _is_negated(self, rel: str, sentence: str) -> bool:
        text = f"{rel} {sentence}".lower()
        return bool(re.search(r"\b(não|nunca|jamais|nem)\b", text))

    def _infer_modal_scope(self, rel: str) -> str:
        rel_c = canon_rel(rel)
        if re.search(r"\b(deve|precisa|pode|poderia|deveria|tem que|há de)\b", rel_c):
            return "deontic"
        if re.search(r"\b(parece|talvez|poderá|seria)\b", rel_c):
            return "epistemic"
        return "unknown"

    def _heuristics_for_candidate(self, cand: CandidateTriple, validation: Optional[TheoryValidation] = None) -> Dict[str, List[str]]:
        used: List[str] = []
        passed: List[str] = []
        failed: List[str] = []

        used.append(f"source_rule:{cand.source_rule}")
        used.append(f"pattern_ud:{cand.pattern_ud}")
        used.append(f"variant_type:{cand.variant_type}")
        used.append(f"theory_mode:{self.config.theory_mode}")

        if self.config.apply_theoretical_rules:
            used.append("apply_theoretical_rules")

        if not self.config.no_attn:
            used.append("attention_enabled")
            used.append(f"heads_mode:{self.config.heads_mode}")
            used.append(f"attn_threshold:{self.config.attn_threshold}")

            if self._pattern_uses_attention_decision(cand.pattern_ud):
                used.append("attn_decision_pattern")
            if self._pattern_uses_attention_rerank(cand.pattern_ud):
                used.append("attn_rerank_pattern")

        if cand.pattern_ud == "cop":
            used.append("extract_cop")
            used.append(f"cop_mode:{self.config.cop_mode}")

        if cand.pattern_ud == "aux:pass":
            used.append("extract_aux_pass")

        if cand.pattern_ud == "modal+verb":
            used.append("extract_modal")

        if "coord" in (cand.pattern_ud or "") or "coord" in (cand.source_rule or ""):
            used.append("extract_coord")

        if cand.pattern_ud in {"verb+obl", "verb+obj[+obl]", "verb+nary-split"}:
            if getattr(self.config, "generate_obl_variants", False):
                used.append("generate_obl_variants")

        if cand.pattern_ud in {"verb+ccomp", "verb+xcomp", "verb+csubj"}:
            used.append("clausal_candidate")

        if cand.source_rule == "nary_split":
            used.append("nary_split")

        if validation is not None:
            for field_name, value in asdict(validation).items():
                if not field_name.startswith("rule_"):
                    continue
                if value is True:
                    passed.append(field_name)
                elif value is False:
                    failed.append(field_name)

            if validation.factuality_label:
                used.append(f"factuality:{validation.factuality_label}")
            if validation.arg1_head_type:
                used.append(f"arg1_head:{validation.arg1_head_type}")
            if validation.arg2_head_type:
                used.append(f"arg2_head:{validation.arg2_head_type}")

        return {
            "heuristics_used": sorted(dict.fromkeys(used)),
            "theory_rules_passed": sorted(dict.fromkeys(passed)),
            "theory_rules_failed": sorted(dict.fromkeys(failed)),
        }

    # -------------------------
    # Geração de candidatos
    # -------------------------
    def _candidate_from_row(
        self,
        row: Dict[str, Any],
        rank: int = 0,
        source_rule: str = "legacy",
        variant_type: str = "base"
    ) -> CandidateTriple:
        pattern = row.get("pattern_ud", "unknown")
    
        arg1 = row.get("arg1", "")
        rel = row.get("rel", "")
        arg2 = row.get("arg2", "")
    
        # Limpeza de spans herdados do extrator legado
        if pattern in {"verb+obj", "verb+obj[+obl]", "verb+obl", "cop", "aux:pass"}:
            arg2 = self._truncate_legacy_arg_text(arg2)
    
        modal_scope = self._infer_modal_scope(rel) if pattern == "modal+verb" else None
        voice = "passive" if pattern == "aux:pass" else None
    
        return CandidateTriple(
            sentence=row.get("sentence", ""),
            arg1=arg1,
            rel=rel,
            arg2=arg2,
            verb_id=row.get("verb_id"),
            subj_id=row.get("subj_id"),
            obj_id=row.get("obj_id"),
            pattern_ud=pattern,
            subj_deprel=row.get("subj_deprel"),
            obj_deprel=row.get("obj_deprel"),
            source_rule=source_rule,
            candidate_rank=rank,
            variant_type=variant_type,
            modal_scope=modal_scope,
            voice=voice,
    
            # -------------------------
            # Campos de atenção
            # -------------------------
            conf_att_subj=row.get("conf_att_subj"),
            conf_att_obj=row.get("conf_att_obj"),
            conf_att=row.get("conf_att"),
    
            best_layer_subj=row.get("best_layer_subj"),
            best_head_subj=row.get("best_head_subj"),
            best_layer_obj=row.get("best_layer_obj"),
            best_head_obj=row.get("best_head_obj"),
    
            att_evidence_subj=row.get("att_evidence_subj", []),
            att_evidence_obj=row.get("att_evidence_obj", []),
    
            selected_heads_subj=row.get("selected_heads_subj", []),
            selected_heads_obj=row.get("selected_heads_obj", []),
        
            heuristics_used=row.get("heuristics_used", []),
            theory_rules_passed=row.get("theory_rules_passed", []),
            theory_rules_failed=row.get("theory_rules_failed", []),
        )

    def _extract_base_svo_candidates(self, sentence: str) -> Tuple[List[CandidateTriple], List[UDTok]]:
        rows, ud = super().extract(sentence)
        cands: List[CandidateTriple] = []
    
        for i, r in enumerate(rows):
            r = dict(r)  # cópia defensiva
    
            src = r.get("pattern_ud", "legacy")
            variant = "extensive" if "[+obl]" in src else ("coord_split" if "coord" in src else "base")
    
            # Limpeza local apenas para padrões herdados do legado
            if src in {"verb+obj", "verb+obj[+obl]", "verb+obl", "cop", "aux:pass"}:
                r["arg2"] = self._truncate_legacy_arg_text(r.get("arg2", ""))
    
            cand = self._candidate_from_row(
                r,
                rank=i,
                source_rule=src,
                variant_type=variant
            )
            cands.append(cand)
    
        return cands, ud

    def _extract_clausal_candidates(
        self,
        sentence: str,
        ud: List[UDTok],
        kids: Dict[int, List[UDTok]]
    ) -> List[CandidateTriple]:
        out: List[CandidateTriple] = []
    
        for v in ud:
            if (v.upos or "").upper() not in {"VERB", "AUX"}:
                continue
    
            v_kids = kids.get(v.id, [])
            subj = next((c for c in v_kids if (c.deprel or "") in {"nsubj", "nsubj:pass", "csubj"}), None)
            if subj is None:
                continue
    
            subj_span = self._expand_arg_extensive(subj.id, ud, kids)
            arg1 = self._span_text(subj_span, ud)
    
            rel, _, _ = self._build_rel_minimal(v, v_kids, None, ud, kids)
            rel, _ = self._decompose_contracted_rel_tokens(rel, "")
    
            for ch in v_kids:
                dep = (ch.deprel or "").lower()
                if dep not in {"ccomp", "xcomp", "csubj"}:
                    continue
    
                arg2 = self._clausal_text(ch.id, ud, kids)
                if not (arg1 and rel and arg2):
                    continue
    
                if dep == "ccomp":
                    patt = "verb+ccomp"
                elif dep == "xcomp":
                    patt = "verb+xcomp"
                else:
                    patt = "verb+csubj"
    
                out.append(
                    CandidateTriple(
                        sentence=sentence,
                        arg1=arg1,
                        rel=rel,
                        arg2=arg2,
                        verb_id=v.id,
                        subj_id=subj.id,
                        obj_id=ch.id,
                        pattern_ud=patt,
                        subj_deprel=subj.deprel or "nsubj",
                        obj_deprel=dep,
                        source_rule="clausal_candidate",
                        candidate_rank=len(out),
                        variant_type="clausal",
                    )
                )
    
        return out

    def _extract_nary_splits(
        self,
        sentence: str,
        ud: List[UDTok],
        kids: Dict[int, List[UDTok]],
        existing: List[CandidateTriple]
    ) -> List[CandidateTriple]:
        """
        Decomposição n-ária conservadora.
        Só gera split para oblíquos com papel relacional plausível.
        """
        out: List[CandidateTriple] = []
    
        for cand in existing:
            if cand.verb_id is None or cand.subj_id is None:
                continue
    
            # só vale a pena tentar split em candidatos estruturais centrais
            if cand.pattern_ud not in {"verb+obj", "verb+obj[+obl]", "verb+obl", "aux:pass", "modal+verb"}:
                continue
    
            v = next((t for t in ud if t.id == cand.verb_id), None)
            subj = next((t for t in ud if t.id == cand.subj_id), None)
            if v is None or subj is None:
                continue
    
            v_kids = kids.get(v.id, [])
            obls = [c for c in v_kids if (c.deprel or "").startswith("obl")]
    
            for obl in obls:
                # não splitar o mesmo complemento já usado como obj_id da própria tripla
                if cand.obj_id is not None and obl.id == cand.obj_id:
                    continue
    
                # só aceita oblíquo central
                if not self._obl_is_core_for_split(obl, v, ud, kids):
                    continue
    
                subj_span = self._expand_arg_extensive(subj.id, ud, kids)
                arg1 = self._span_text(subj_span, ud)
    
                rel, _, prep_excl = self._build_rel_minimal(v, v_kids, obl.id, ud, kids)
                rel = self._attach_governed_preposition_to_rel(rel, obl, kids)
                rel, _ = self._decompose_contracted_rel_tokens(rel, "")
    
                obl_span = self._expand_arg_extensive(obl.id, ud, kids)
                arg2 = self._span_text([i for i in obl_span if i not in prep_excl], ud)
    
                if not (arg1 and rel and arg2):
                    continue
    
                out.append(
                    CandidateTriple(
                        sentence=sentence,
                        arg1=arg1,
                        rel=rel,
                        arg2=arg2,
                        verb_id=v.id,
                        subj_id=subj.id,
                        obj_id=obl.id,
                        pattern_ud="verb+nary-split",
                        subj_deprel=subj.deprel or "nsubj",
                        obj_deprel=obl.deprel or "obl",
                        source_rule="nary_split",
                        candidate_rank=len(out),
                        variant_type="nary_split",
                    )
                )
    
        return out

    def _generate_all_candidates(
        self,
        sentence: str,
        ud: Optional[List[UDTok]] = None,
        kids: Optional[Dict[int, List[UDTok]]] = None
    ) -> Tuple[List[CandidateTriple], List[UDTok], Dict[int, List[UDTok]]]:
        base_svo, parsed_ud = self._extract_base_svo_candidates(sentence)
    
        ud = ud or parsed_ud
        kids = kids or self._children_index(ud)
    
        base_obl = self._extract_base_verb_obl_candidates(sentence, ud, kids)
    
        cands = list(base_svo) + list(base_obl)
        cands.extend(self._extract_clausal_candidates(sentence, ud, kids))
        cands.extend(self._extract_nary_splits(sentence, ud, kids, cands))
    
        # deduplicação inicial
        seen = set()
        uniq: List[CandidateTriple] = []
        for c in cands:
            key = (canon_arg(c.arg1), canon_rel(c.rel), canon_arg(c.arg2), c.pattern_ud)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
    
        return uniq, ud, kids

    # -------------------------
    # Validação teórica
    # -------------------------
    def _validate_E1(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if not (cand.arg1 and cand.rel and cand.arg2):
            return False
    
        if cand.verb_id is None:
            return False
    
        vt = next((t for t in ud if t.id == cand.verb_id), None)
        if vt is None:
            return False
    
        rel_c = canon_rel(cand.rel)
        v_text = canon_rel(vt.text or "")
        v_lemma = canon_rel(vt.lemma or "")
        child_tokens = kids.get(vt.id, [])
    
        # 1) REL deve conter ao menos alguma realização verbal plausível
        rel_has_surface_verb = any(
            x and re.search(rf"\b{re.escape(x)}\b", rel_c)
            for x in {v_text, v_lemma}
            if x
        )
    
        # 2) REL pode ser sustentada por auxiliar/cópula/particípio ligado ao verbo
        verbal_support = False
        for ch in child_tokens:
            ch_text = canon_rel(ch.text or "")
            ch_lemma = canon_rel(ch.lemma or "")
            if ch.deprel.startswith("aux") or ch.deprel == "cop":
                if any(
                    x and re.search(rf"\b{re.escape(x)}\b", rel_c)
                    for x in {ch_text, ch_lemma}
                    if x
                ):
                    verbal_support = True
                    break
    
        # 3) padrões copulares e passivos exigem critério menos ingênuo
        if cand.pattern_ud == "cop":
            has_copula_form = bool(re.search(r"\b(é|foi|era|são|ser|estar|está|estavam|estiveram)\b", rel_c))
            return has_copula_form and bool(cand.arg1) and bool(cand.arg2)
    
        if cand.pattern_ud == "aux:pass":
            has_passive_shape = bool(re.search(r"\b(foi|foram|era|eram|sido|sendo)\b", rel_c)) or verbal_support
            return has_passive_shape and bool(cand.arg1) and bool(cand.arg2)
    
        # 4) casos clausais: basta haver núcleo verbal e argumento clausal
        if cand.pattern_ud in {"verb+ccomp", "verb+xcomp", "verb+csubj"}:
            return (rel_has_surface_verb or verbal_support) and bool(cand.arg2)
    
        # 5) caso geral: vínculo estrutural mínimo
        has_subject_link = cand.subj_id is not None or cand.subj_deprel in {"nsubj", "nsubj:pass", "csubj"}
        has_object_link = cand.obj_id is not None or cand.obj_deprel not in {"", None, "unknown"}
    
        return (rel_has_surface_verb or verbal_support) and has_subject_link and has_object_link

    def _validate_E2(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        # heurística leve: só verifica Number/Person quando ambos existirem
        subj = next((t for t in ud if t.id == cand.subj_id), None)
        verb = next((t for t in ud if t.id == cand.verb_id), None)
        if subj is None or verb is None:
            return True
        sf = getattr(subj, "feats", None) or {}
        vf = getattr(verb, "feats", None) or {}
        for feat in ("Number", "Person"):
            if sf.get(feat) and vf.get(feat) and sf.get(feat) != vf.get(feat):
                return False
        return True

    def _validate_E3(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if not cand.arg1 or not cand.arg2:
            return False
        if self._detect_nonessential_relative_clause(cand.arg1, ud, kids):
            return False
        if self._detect_nonessential_relative_clause(cand.arg2, ud, kids):
            return False
        return True

    def _validate_E4(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        obj = next((t for t in ud if t.id == cand.obj_id), None)
        if obj is None:
            return True
        case_children = [canon_rel(c.text) for c in kids.get(obj.id, []) if c.deprel == "case"]
        if not case_children:
            return True
        rel_c = canon_rel(cand.rel)
        return any(p in rel_c for p in case_children)

    def _validate_E4_1(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        bad = re.search(r"\b(do|da|dos|das|no|na|nos|nas|ao|aos|à|às)\b", cand.rel.lower())
        # se a contração ficou em REL é aceitável apenas em modo não estrito
        return not bad if self.config.strict_theory else True

    def _validate_E4_2(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        loc = self._detect_prepositional_locution(cand.rel)
        if loc is None:
            return True
        return canon_rel(loc) in canon_rel(cand.rel)

    def _validate_E5(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        Ao menos um argumento deve ter âncora nominal/pronominal clara.
        Rebaixa casos de pronome relativo isolado como 'que'.
        """
    
        def head_type(tok_id: Optional[int], fallback_text: str = "") -> str:
            if tok_id is None:
                txt = canon_arg(fallback_text)
                if txt.startswith("que " ) or txt == "que":
                    return "PRON"
                return "OTHER"
    
            tok = next((t for t in ud if t.id == tok_id), None)
            if tok is None:
                txt = canon_arg(fallback_text)
                if txt.startswith("que ") or txt == "que":
                    return "PRON"
                return "OTHER"
    
            up = (tok.upos or "").upper()
            if up in {"NOUN", "PROPN", "PRON", "ADJ", "VERB", "AUX"}:
                return "CLAUSE" if up in {"VERB", "AUX"} else up
            return "OTHER"
    
        a1_type = head_type(cand.subj_id, cand.arg1)
        a2_type = head_type(cand.obj_id, cand.arg2)
    
        a1 = a1_type in {"NOUN", "PROPN", "PRON"}
        a2 = a2_type in {"NOUN", "PROPN", "PRON", "CLAUSE"}
    
        # Evita aceitar sujeito relativo muito pobre como âncora principal
        if canon_arg(cand.arg1) == "que" and a1_type == "PRON":
            return a2
    
        return a1 or a2

    def _validate_E6(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if cand.variant_type == "nary_split":
            return True
        if cand.pattern_ud == "verb+obj[+obl]":
            return True
        return True

    def _validate_E7(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        return True if "coord" in cand.pattern_ud or cand.source_rule.endswith("coordination_split") or cand.source_rule == "legacy" else True

    def _validate_E8(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        clausal_arg = cand.pattern_ud in {"verb+ccomp", "verb+xcomp", "verb+csubj"}
        if not self.config.e8_allow_clausal_args:
            return not clausal_arg
        return True

    def _validate_E9(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        loc = self._detect_prepositional_locution(cand.rel)
        return True if loc is None else canon_rel(loc) in canon_rel(cand.rel)

    def _validate_S1(
        self,
        cand: CandidateTriple,
        ud: List[UDTok],
        kids: Dict[int, List[UDTok]]
    ) -> Tuple[bool, str]:
        """
        S1: factualidade e implicação.
        - negação preservada continua válida;
        - modal é válido, mas rotulado como modalized;
        - reported speech/belief é anotado;
        - condicional só bloqueia quando há evidência local real.
        """
        neg = self._is_negated(cand.rel, cand.sentence)
        cond = self._detect_conditional_scope_for_verb(cand.verb_id, ud, kids)
        rep, verb = self._detect_reported_belief_or_speech(cand.rel)
    
        if neg and getattr(self.config, "s1_keep_negation", True):
            return True, "negated"
    
        if cand.pattern_ud == "modal+verb":
            return True, "modalized"
    
        if rep:
            if verb in set(getattr(self.config, "belief_verbs", [])) and getattr(self.config, "s1_block_reported_belief", False):
                return False, "reported"
            return True, "reported"
    
        if cond and getattr(self.config, "s1_block_conditionals", True):
            return False, "conditional"
    
        return True, "asserted"

    def _validate_S2(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if self.config.s2_span_policy == "both":
            return True
        if self.config.s2_span_policy == "minimal":
            return cand.variant_type in {"base", "minimal", "clausal"}
        if self.config.s2_span_policy == "extensive":
            return cand.variant_type in {"extensive", "nary_split", "coord_split", "clausal", "base"}
        return True

    def _validate_S3(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        S3: evitar complexidade/aninhamento indevido.
        Penaliza:
        - verb+obj[+obl] quando o obl anexado é periférico;
        - nary_split quando o obl destacado não é central;
        - argumentos com forte sinal de material parentético/metalinguístico.
        """
        v = next((t for t in ud if t.id == cand.verb_id), None) if cand.verb_id is not None else None
        obj = next((t for t in ud if t.id == cand.obj_id), None) if cand.obj_id is not None else None
    
        # 1) split n-ário indevido
        if cand.source_rule == "nary_split" and v is not None and obj is not None:
            return self._obl_is_core_for_split(obj, v, ud, kids)
    
        # 2) expansão excessiva com obl periférico
        if cand.pattern_ud == "verb+obj[+obl]" and v is not None:
            v_kids = kids.get(v.id, [])
            obls = [c for c in v_kids if (c.deprel or "").startswith("obl")]
            # se existir obl anexado mas nenhum for central para attachment, falha
            if obls and not any(self._obl_is_core_for_attachment(obl, v, ud, kids) for obl in obls):
                return False
    
        # 3) sinais textuais de complexidade excessiva
        arg2 = canon_arg(cand.arg2)
        bad_signals = [
            " em latim ",
            " em inglês ",
            " em português ",
            " abreviado ",
            " abreviados ",
            " pós nominais ",
            " pos nominais ",
        ]
        padded = f" {arg2} "
        if any(sig in padded for sig in bad_signals):
            return False
    
        if "(" in cand.arg2 or ")" in cand.arg2:
            return False
    
        return True

    def _validate_S4(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        if cand.pattern_ud == "cop":
            return bool(re.search(r"\b(é|foi|era|são|está|estão|ser|estar)\b", cand.rel.lower()))
        if cand.pattern_ud == "aux:pass":
            return cand.voice == "passive" or bool(re.search(r"\b(foi|foram|sido|sendo)\b", cand.rel.lower()))
        return True

    def _validate_S5(self, cand: CandidateTriple, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        # Nesta implementação não há inferência externa geradora de triplas.
        return True

    def _validate_candidate_theory(self, sentence: str, ud: List[UDTok], kids: Dict[int, List[UDTok]], cand: CandidateTriple) -> TheoryValidation:
        tv = TheoryValidation()
        tv.arg1_head_type = self._classify_argument_head_type(cand.arg1, ud)
        tv.arg2_head_type = self._classify_argument_head_type(cand.arg2, ud)

        checks = {
            "E1": self._validate_E1,
            "E2": self._validate_E2,
            "E3": self._validate_E3,
            "E4": self._validate_E4,
            "E4_1": self._validate_E4_1,
            "E4_2": self._validate_E4_2,
            "E5": self._validate_E5,
            "E6": self._validate_E6,
            "E7": self._validate_E7,
            "E8": self._validate_E8,
            "E9": self._validate_E9,
            "S2": self._validate_S2,
            "S3": self._validate_S3,
            "S4": self._validate_S4,
            "S5": self._validate_S5,
        }
        for rule, fn in checks.items():
            if getattr(self.config, f"apply_{rule}", True):
                setattr(tv, f"rule_{rule}", bool(fn(cand, ud, kids)))

        if self.config.apply_S1:
            ok_s1, label = self._validate_S1(cand, ud, kids)
            tv.rule_S1 = bool(ok_s1)
            tv.factuality_label = label

        critical = ["E1", "E4", "E5", "S1", "S4", "S5"]
        noncritical = ["E2", "E3", "E4_1", "E4_2", "E6", "E7", "E8", "E9", "S2", "S3"]

        all_results = {}
        for rule in critical + noncritical:
            val = getattr(tv, f"rule_{rule}", None)
            if val is not None:
                all_results[rule] = bool(val)

        tv.critical_violations = [r for r in critical if all_results.get(r) is False]
        tv.warnings = [r for r in noncritical if all_results.get(r) is False]
        tv.notes = [f"factuality={tv.factuality_label}", f"arg1_head={tv.arg1_head_type}", f"arg2_head={tv.arg2_head_type}"]
        n_total = len(all_results)
        n_ok = sum(1 for v in all_results.values() if v)
        tv.theory_score = float(n_ok / n_total) if n_total else 1.0
        tv.theory_valid = len(tv.critical_violations) == 0
        tv.theory_blocked = self.config.theory_mode == "filter" and not tv.theory_valid
        return tv

    def _apply_theory_policy(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.config.theory_mode in {"off", "annotate"}:
            return rows
        return [r for r in rows if not r.get("theory_blocked", False)]

    def _finalize_output_row(self, cand: CandidateTriple, validation: TheoryValidation) -> Dict[str, Any]:
        row = asdict(cand)
        row.update(asdict(validation))
        row["critical_violations"] = ";".join(validation.critical_violations)
        row["theory_notes"] = ";".join(validation.notes)
        row["warnings"] = ";".join(validation.warnings)
        # -------------------------
        # Campos de atenção
        # -------------------------
        row["conf_att_subj"] = getattr(cand, "conf_att_subj", None)
        row["conf_att_obj"] = getattr(cand, "conf_att_obj", None)
        row["conf_att"] = getattr(cand, "conf_att", None)
        row["att_evidence_obj"] = getattr(cand, "att_evidence_obj", [])
        row["att_evidence_subj"] = getattr(cand, "att_evidence_subj", [])
        row["best_layer_obj"] = getattr(cand, "best_layer_obj", None)
        row["best_head_obj"] = getattr(cand, "best_head_obj", None)
        row["best_head_score_obj"] = getattr(cand, "best_head_score_obj", None)
        row["best_layer_subj"] = getattr(cand, "best_layer_subj", None)
        row["best_head_subj"] = getattr(cand, "best_head_subj", None)
        row["best_head_score_subj"] = getattr(cand, "best_head_score_subj", None)
        row["selected_heads_obj"] = getattr(cand, "selected_heads_obj", [])
        row["selected_heads_subj"] = getattr(cand, "selected_heads_subj", [])
        row["attn_decision_pattern"] = self._pattern_uses_attention_decision(row.get("pattern_ud", "unknown"))
        row["attn_threshold"] = self.config.attn_threshold if not self.config.no_attn else None

        heur_info = self._heuristics_for_candidate(cand, validation)
        row["heuristics_used"] = heur_info["heuristics_used"]
        row["heuristics_used_txt"] = " | ".join(heur_info["heuristics_used"])
        row["theory_rules_passed"] = heur_info["theory_rules_passed"]
        row["theory_rules_passed_txt"] = " | ".join(heur_info["theory_rules_passed"])
        row["theory_rules_failed"] = heur_info["theory_rules_failed"]
        row["theory_rules_failed_txt"] = " | ".join(heur_info["theory_rules_failed"])
        return row

    def extract(self, sentence: str) -> Tuple[List[Dict[str, Any]], List[UDTok]]:
        cands, ud, kids = self._generate_all_candidates(sentence)
    
        rows: List[Dict[str, Any]] = []
        seen = set()
    
        for cand in cands:
            validation = (
                self._validate_candidate_theory(sentence, ud, kids, cand)
                if self.config.apply_theoretical_rules
                else TheoryValidation()
            )
    
            row = self._finalize_output_row(cand, validation)
    
            key = (
                canon_arg(row["arg1"]),
                canon_rel(row["rel"]),
                canon_arg(row["arg2"]),
                row.get("pattern_ud"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    
        # primeiro aplica a política teórica
        rows = self._apply_theory_policy(rows)
    
        # depois aplica reranking por atenção, sem poda dura
        rows = self._apply_attention_rerank(rows)
    
        return rows, ud

    def _token_in_parenthetical(self, tok_id: Optional[int], ud: List[UDTok]) -> bool:
        if tok_id is None:
            return False
        idx = None
        for i, t in enumerate(ud):
            if t.id == tok_id:
                idx = i
                break
        if idx is None:
            return False
    
        left_text = " ".join(t.text for t in ud[max(0, idx - 6):idx])
        right_text = " ".join(t.text for t in ud[idx:min(len(ud), idx + 7)])
    
        ctx = f"{left_text} {right_text}"
        return "(" in ctx or ")" in ctx


    def _is_marginal_obl_for_nary_split(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        Bloqueia oblíquos marginais, parentéticos, metalinguísticos ou
        adjuntos introdutórios que não devem virar tripla binária.
        """
        txt = canon_arg(obl_tok.text or "")
        lem = canon_arg(obl_tok.lemma or "")
    
        # Parentéticos e incisos
        if self._token_in_parenthetical(obl_tok.id, ud):
            return True
    
        # Muito curto ou pouco informativo
        if txt in {"latim", "português", "inglês"}:
            return True
    
        # Casos metalinguísticos comuns
        local_span_ids = self._expand_arg_extensive(obl_tok.id, ud, kids)
        local_span_text = canon_arg(self._span_text(local_span_ids, ud))
        bad_phrases = {
            "em latim",
            "em inglês",
            "em português",
            "abreviados",
            "abreviado",
            "pós nominais",
            "pos nominais",
        }
        if any(bp in local_span_text for bp in bad_phrases):
            return True
    
        # Adjuntos temporais/discursivos introdutórios antes do verbo
        if obl_tok.id < verb_tok.id:
            case_children = [c for c in kids.get(obl_tok.id, []) if (c.deprel or "").startswith("case")]
            case_text = canon_arg(" ".join(c.text for c in sorted(case_children, key=lambda x: x.id)))
            intro_markers = {"em", "durante", "após", "antes", "desde"}
            if case_text in intro_markers:
                return True
    
        return False
    
    
    def _obl_is_core_for_split(self, obl_tok: UDTok, verb_tok: UDTok, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> bool:
        """
        Mantém apenas oblíquos com chance razoável de valor relacional central.
        """
        if self._is_marginal_obl_for_nary_split(obl_tok, verb_tok, ud, kids):
            return False
    
        span_ids = self._expand_arg_extensive(obl_tok.id, ud, kids)
        span_text = canon_arg(self._span_text(span_ids, ud))
    
        # Aceita alguns casos mais plausíveis
        good_patterns = [
            r"\b(em|no|na|nos|nas)\s+\d{4}\b",          # em 1948
            r"\b(para|a)\s+[a-zà-ÿ]",                   # para Memphis / a Roma
            r"\b(em|no|na)\s+[a-zà-ÿ]",                 # em Salvador / no Brasil
            r"\b(de|do|da)\s+[a-zà-ÿ]",                 # origem/associação
        ]
        return any(re.search(p, span_text) for p in good_patterns)

    def _expand_clause_span(self, head_id: int, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> List[int]:
        """
        Expande uma oração subordinada de forma conservadora:
        inclui cabeça verbal, sujeito, objeto, marcadores e complementos essenciais.
        """
        keep = set()
    
        def rec(tok_id: int):
            if tok_id in keep:
                return
            keep.add(tok_id)
            for ch in kids.get(tok_id, []):
                dep = (ch.deprel or "").lower()
    
                if dep in {
                    "nsubj", "nsubj:pass", "obj", "iobj",
                    "ccomp", "xcomp", "obl", "obl:arg",
                    "mark", "advmod", "compound:prt", "fixed",
                    "det", "amod", "nummod", "flat", "flat:name",
                    "compound", "case"
                }:
                    rec(ch.id)
    
        rec(head_id)
        return self._prune_nonessential_material(sorted(keep), ud, kids)


    def _clausal_text(self, head_id: int, ud: List[UDTok], kids: Dict[int, List[UDTok]]) -> str:
        ids = self._expand_clause_span(head_id, ud, kids)
        txt = self._span_text(ids, ud).strip()
    
        # preserva "que" inicial, quando houver mark
        head = next((t for t in ud if t.id == head_id), None)
        if head is not None:
            marks = [c for c in kids.get(head.id, []) if (c.deprel or "").startswith("mark")]
            mark_txt = " ".join(c.text for c in sorted(marks, key=lambda x: x.id)).strip()
            if mark_txt and not canon_arg(txt).startswith(canon_arg(mark_txt)):
                txt = f"{mark_txt} {txt}".strip()
    
        return txt

    def _extract_base_verb_obl_candidates(
        self,
        sentence: str,
        ud: List[UDTok],
        kids: Dict[int, List[UDTok]]
    ) -> List[CandidateTriple]:
        """
        Gera candidatos base do tipo verbo + complemento oblíquo,
        mas somente quando o obl tiver papel central plausível.
        Ex.: 'David viaja para outro país.'
        Não deve gerar casos introdutórios como:
        'Em fevereiro de 2013, Beyoncé disse ...'
        """
        out: List[CandidateTriple] = []
    
        for v in ud:
            if (v.upos or "").upper() not in {"VERB", "AUX"}:
                continue
    
            v_kids = kids.get(v.id, [])
    
            subj = next((c for c in v_kids if (c.deprel or "") in {"nsubj", "nsubj:pass", "csubj"}), None)
            if subj is None:
                continue
    
            # se já houver objeto direto, deixa para outro padrão
            has_obj = any((c.deprel or "").startswith("obj") for c in v_kids)
            if has_obj:
                continue
    
            obls = [c for c in v_kids if (c.deprel or "").startswith("obl")]
            if not obls:
                continue
    
            for obl in obls:
                # NOVO: só aceita obl central
                if not self._obl_is_core_for_attachment(obl, v, ud, kids):
                    continue
    
                subj_span = self._expand_arg_extensive(subj.id, ud, kids)
                arg1 = self._span_text(subj_span, ud)
    
                rel, _, prep_excl = self._build_rel_minimal(v, v_kids, obl.id, ud, kids)
                rel = self._attach_governed_preposition_to_rel(rel, obl, kids)
                rel, _ = self._decompose_contracted_rel_tokens(rel, "")
    
                obl_span = self._expand_arg_extensive(obl.id, ud, kids)
                arg2 = self._span_text([i for i in obl_span if i not in prep_excl], ud)
    
                if not (arg1 and rel and arg2):
                    continue
    
                out.append(
                    CandidateTriple(
                        sentence=sentence,
                        arg1=arg1,
                        rel=rel,
                        arg2=arg2,
                        verb_id=v.id,
                        subj_id=subj.id,
                        obj_id=obl.id,
                        pattern_ud="verb+obl",
                        subj_deprel=subj.deprel or "nsubj",
                        obj_deprel=obl.deprel or "obl",
                        source_rule="base_verb_obl",
                        candidate_rank=len(out),
                        variant_type="base",
                    )
                )
    
        return out


# -----------------------------------------------------------------------------
# Gold enrichment teórico
# -----------------------------------------------------------------------------
def infer_surface_pattern_from_gold(extractor: OpenIEExtractorBIA, sentence: str, g: Dict[str, Any]) -> Dict[str, Any]:
    base = infer_surface_pattern_from_gold_legacy(extractor, sentence, g)
    # perfil teórico leve do gold
    sent = sentence.lower()
    rel = str(g.get("rel", "")).lower()
    arg2 = str(g.get("arg2", "")).lower()
    gold_theory_profile = {
        "gold_rule_E8": bool(re.search(r"\bque\b", arg2) or re.search(r"\bque\b", rel)),
        "gold_rule_S1_negated": bool(re.search(r"\b(não|nunca|jamais)\b", sent + " " + rel)),
        "gold_rule_S1_modal": bool(re.search(r"\b(pode|deve|precisa|deveria|poderia)\b", sent + " " + rel)),
        "gold_rule_E7_coord": bool(re.search(r"\b(e|ou)\b", arg2)),
        "gold_rule_E6_nary_like": bool(re.search(r"\b(em|para|com|de)\b", arg2)) and len(arg2.split()) >= 3,
        "gold_rule_cop_like": canon_rel(rel) in {"é", "foi", "era", "são", "está", "estão", "ser", "estar"},
    }
    base["gold_theory_profile"] = json.dumps(gold_theory_profile, ensure_ascii=False)
    base.update(gold_theory_profile)
    return base


def infer_gold_patterns(extractor: OpenIEExtractorBIA, gold_rows: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    out = []
    for item in gold_rows:
        row = dict(item)
        row.update(infer_surface_pattern_from_gold(extractor, row["sentence"], row))
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Flatten / relatórios adicionais
# -----------------------------------------------------------------------------
def flatten_pred_rows(preds_by_sent: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for sent_preds in preds_by_sent:
        out.extend(sent_preds)
    return out


def flatten_gold_rows(gold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(gold_rows)


def summarize_by_rule(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = [f"rule_E{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]] + ["rule_E4_1", "rule_E4_2"] + [f"rule_S{i}" for i in [1, 2, 3, 4, 5]]
    ordered_rules = ["rule_E1", "rule_E2", "rule_E3", "rule_E4", "rule_E4_1", "rule_E4_2", "rule_E5", "rule_E6", "rule_E7", "rule_E8", "rule_E9", "rule_S1", "rule_S2", "rule_S3", "rule_S4", "rule_S5"]
    rows = []
    for rule in ordered_rules:
        vals = [r.get(rule) for r in pred_rows if r.get(rule) is not None]
        if not vals:
            continue
        passed = sum(bool(v) for v in vals)
        failed = sum(not bool(v) for v in vals)
        rows.append({"rule": rule, "n": len(vals), "passed": passed, "failed": failed, "pass_rate": passed / len(vals) if vals else 0.0})
    return rows


def summarize_by_factuality(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    c = Counter(r.get("factuality_label", "unknown") for r in pred_rows)
    n = sum(c.values()) or 1
    return [{"factuality_label": k, "count": v, "rate": v / n} for k, v in sorted(c.items())]


def summarize_by_variant(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    c = Counter((r.get("variant_type", "unknown"), bool(r.get("theory_valid", False))) for r in pred_rows)
    rows = []
    for (variant, valid), count in sorted(c.items()):
        rows.append({"variant_type": variant, "theory_valid": valid, "count": count})
    return rows


def summarize_by_pattern_and_rule(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patterns = sorted(set(r.get("pattern_ud", "unknown") for r in pred_rows))
    rules = ["rule_E1", "rule_E2", "rule_E3", "rule_E4", "rule_E4_1", "rule_E4_2", "rule_E5", "rule_E6", "rule_E7", "rule_E8", "rule_E9", "rule_S1", "rule_S2", "rule_S3", "rule_S4", "rule_S5"]
    rows = []
    for pat in patterns:
        subset = [r for r in pred_rows if r.get("pattern_ud") == pat]
        for rule in rules:
            vals = [r.get(rule) for r in subset if r.get(rule) is not None]
            if not vals:
                continue
            rows.append({
                "pattern_ud": pat,
                "rule": rule,
                "n": len(vals),
                "passed": sum(bool(v) for v in vals),
                "failed": sum(not bool(v) for v in vals),
                "pass_rate": sum(bool(v) for v in vals) / len(vals),
            })
    return rows


# -----------------------------------------------------------------------------
# Avaliação
# -----------------------------------------------------------------------------
def evaluate_dataset(preds_by_sent: List[List[Dict[str, Any]]], gold_rows: List[Dict[str, Any]], config: Config) -> Dict[str, Any]:
    base = evaluate_dataset_legacy(preds_by_sent, gold_rows, config)
    pred_rows = flatten_pred_rows(preds_by_sent)

    theory_valid_count = sum(bool(r.get("theory_valid", False)) for r in pred_rows)
    theory_blocked_count = sum(bool(r.get("theory_blocked", False)) for r in pred_rows)

    base["per_rule_summary"] = summarize_by_rule(pred_rows)
    base["per_factuality_summary"] = summarize_by_factuality(pred_rows)
    base["per_variant_type_summary"] = summarize_by_variant(pred_rows)
    base["per_pattern_and_rule_summary"] = summarize_by_pattern_and_rule(pred_rows)
    base["n_predictions_total"] = len(pred_rows)
    base["n_predictions_theory_valid"] = theory_valid_count
    base["n_predictions_theory_blocked"] = theory_blocked_count
    base["theory_valid_rate"] = theory_valid_count / len(pred_rows) if pred_rows else 0.0
    return base


# -----------------------------------------------------------------------------
# Execução de experimento
# -----------------------------------------------------------------------------
def run_experiment(config: Config, gold_path: str, output_dir: str, dataset_name: str = "dataset") -> Dict[str, Any]:
    from tqdm.auto import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl = output_dir / f"{dataset_name}_predictions_enriched.jsonl"
    predictions_csv = output_dir / f"{dataset_name}_predictions_enriched.csv"
    gold_csv = output_dir / f"{dataset_name}_gold_enriched.csv"
    triples_table_csv = output_dir / f"{dataset_name}_triples_table.csv"
    per_pattern_metrics_csv = output_dir / f"{dataset_name}_per_pattern_metrics.csv"
    per_rule_summary_csv = output_dir / f"{dataset_name}_per_rule_summary.csv"
    per_factuality_summary_csv = output_dir / f"{dataset_name}_per_factuality_summary.csv"
    per_variant_type_summary_csv = output_dir / f"{dataset_name}_per_variant_type_summary.csv"
    per_pattern_and_rule_summary_csv = output_dir / f"{dataset_name}_per_pattern_and_rule_summary.csv"

    head_usage_overall_csv = output_dir / f"{dataset_name}_head_usage_overall.csv"
    head_usage_by_pattern_csv = output_dir / f"{dataset_name}_head_usage_by_pattern.csv"
    head_usage_summary_json = output_dir / f"{dataset_name}_head_usage_summary.json"

    selected_heads_json = output_dir / f"{dataset_name}_selected_heads.json"

    metrics_path = output_dir / f"{dataset_name}_metrics.json"
    summary_path = output_dir / f"{dataset_name}_summary.json"

    extractor = OpenIEExtractorBIA(config=config, verbose=True)
    gold_rows = read_gold_jsonl(gold_path)
    gold_rows = infer_gold_patterns(extractor, gold_rows, config)

    preds_by_sent: List[List[Dict[str, Any]]] = []
    for item in tqdm(gold_rows, desc=f"Extraindo triplas [{dataset_name}]", unit="sent"):
        preds, _ = extractor.extract(item["sentence"])
        preds_by_sent.append(preds)

    metrics = evaluate_dataset(preds_by_sent, gold_rows, config)

    pred_jsonl_rows = [
        {"sentence": item["sentence"], "pred": preds}
        for item, preds in zip(gold_rows, preds_by_sent)
    ]

    write_jsonl(predictions_jsonl, pred_jsonl_rows)
    write_csv(predictions_csv, flatten_pred_rows(preds_by_sent))
    write_csv(gold_csv, flatten_gold_rows(gold_rows))
    write_csv(triples_table_csv, build_triples_table_legacy(preds_by_sent, gold_rows, config))
    write_csv(per_pattern_metrics_csv, metrics.get("per_pattern", []))
    write_csv(per_rule_summary_csv, metrics.get("per_rule_summary", []))
    write_csv(per_factuality_summary_csv, metrics.get("per_factuality_summary", []))
    write_csv(per_variant_type_summary_csv, metrics.get("per_variant_type_summary", []))
    write_csv(per_pattern_and_rule_summary_csv, metrics.get("per_pattern_and_rule_summary", []))

    head_usage = aggregate_head_usage(preds_by_sent)
    write_csv(head_usage_overall_csv, head_usage.get("overall", []))
    write_csv(head_usage_by_pattern_csv, head_usage.get("by_pattern", []))
    head_usage_summary_json.write_text(
        json.dumps(head_usage, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    selected_heads = dict(getattr(extractor, "heads_meta", {}))
    selected_heads["bert_model"] = config.bert_model
    selected_heads["theory_mode"] = config.theory_mode
    selected_heads["apply_theoretical_rules"] = config.apply_theoretical_rules
    selected_heads_json.write_text(
        json.dumps(selected_heads, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary = {
        "dataset": dataset_name,
        "gold_path": str(gold_path),
        "output_dir": str(output_dir),
        "bert_model": config.bert_model,
        "theory_mode": config.theory_mode,
        "apply_theoretical_rules": config.apply_theoretical_rules,
        "TP": metrics.get("TP"),
        "FP": metrics.get("FP"),
        "FN": metrics.get("FN"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "theory_valid_rate": metrics.get("theory_valid_rate"),
        "n_predictions_total": metrics.get("n_predictions_total"),
        "n_predictions_theory_valid": metrics.get("n_predictions_theory_valid"),
        "n_predictions_theory_blocked": metrics.get("n_predictions_theory_blocked"),
    }

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "metrics": metrics,
        "summary": summary,
        "preds_by_sent": preds_by_sent,
        "gold_rows": gold_rows,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "predictions_csv": str(predictions_csv),
        "predictions_jsonl": str(predictions_jsonl),
        "gold_csv": str(gold_csv),
        "triples_table_csv": str(triples_table_csv),
        "per_rule_summary_csv": str(per_rule_summary_csv),
        "per_factuality_summary_csv": str(per_factuality_summary_csv),
        "per_variant_type_summary_csv": str(per_variant_type_summary_csv),
        "per_pattern_and_rule_summary_csv": str(per_pattern_and_rule_summary_csv),
        "per_pattern_metrics_csv": str(per_pattern_metrics_csv),
        "selected_heads_json": str(selected_heads_json),
        "head_usage_overall_csv": str(head_usage_overall_csv),
        "head_usage_by_pattern_csv": str(head_usage_by_pattern_csv),
        "head_usage_summary_json": str(head_usage_summary_json),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenIE-PT v11 com validação teórica explícita")
    p.add_argument("--gold", required=True, help="JSONL gold com campo sentence")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--bert-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--bosque", default=None)
    p.add_argument("--heads-mode", default="rank")
    p.add_argument("--top-k-heads", type=int, default=10)
    p.add_argument("--attn-threshold", type=float, default=0.0)
    p.add_argument("--no-attn", action="store_true")
    p.add_argument("--theory-mode", choices=["off", "annotate", "filter"], default="filter")
    p.add_argument("--s2-span-policy", choices=["minimal", "extensive", "both"], default="both")
    p.add_argument("--strict-theory", action="store_true")
    p.add_argument("--block-reported-belief", action="store_true")
    p.add_argument("--no-e8-clausal", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = Config(
        bert_model=args.bert_model,
        bosque_path=args.bosque,
        heads_mode=args.heads_mode,
        top_k_heads=args.top_k_heads,
        attn_threshold=args.attn_threshold,
        no_attn=args.no_attn,
        theory_mode=args.theory_mode,
        s2_span_policy=args.s2_span_policy,
        strict_theory=args.strict_theory,
        s1_block_reported_belief=args.block_reported_belief,
        e8_allow_clausal_args=not args.no_e8_clausal,
    )
    run_experiment(config=config, gold_path=args.gold, output_dir=args.output_dir, dataset_name=args.dataset_name)
    logger.info("✅ Experimento concluído")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
