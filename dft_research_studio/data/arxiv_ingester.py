from __future__ import annotations
import hashlib, json, logging, os, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARXIV_KEYWORDS = [
    "density functional theory benchmark",
    "exchange-correlation functional DFT",
    "dispersion correction DFT",
    "noncovalent interactions DFT",
    "hybrid functional benchmark thermochemistry",
]

EXTRACTION_PROMPT = """You are a quantum chemistry knowledge graph expert.
Extract structured knowledge from this DFT paper excerpt.

PAPER EXCERPT:
{text}

Respond ONLY with valid JSON:
{{
  "nodes": [
    {{"node_id":"<id>","label":"<Functional|Benchmark|ValidationResult|FailureMode|Paper|DispersionCorrection>","value":"<numeric or null>","unit":"<unit or null>","rung":"<GGA|meta-GGA|hybrid|double-hybrid or null>"}}
  ],
  "relationships": [
    {{"source_id":"<id>","target_id":"<id>","relationship_type":"<VERB_CAPS>","paper_id":"<paper_id>"}}
  ],
  "paper_id": "<AuthorYear>"
}}
Rules:
- ValidationResult node_ids: VR_<BENCHMARK>_<FUNCTIONAL>
- Only extract entities with quantitative evidence
- relationship_type must start with VERB in CAPS"""


@dataclass
class ArXivPaper:
    arxiv_id:  str
    title:     str
    abstract:  str
    authors:   List[str]
    pdf_url:   str
    published: str
    paper_id:  str = ""

    def __post_init__(self):
        if not self.paper_id:
            first = self.authors[0].split()[-1] if self.authors else "Unknown"
            self.paper_id = f"{first}{self.published[:4]}"


@dataclass
class ExtractionResult:
    paper:         ArXivPaper
    nodes:         List[Dict]
    relationships: List[Dict]
    new_node_count:int = 0
    new_rel_count: int = 0
    status:        str = "pending"


class ArXivIngester:
    """
    Auto-KGR Paper Ingestion Pipeline.
    Automatically queries ArXiv, extracts KG nodes/relationships,
    and merges them into the existing knowledge graph.
    Implements the 'Auto' in Auto-KGR.
    """

    ARXIV_API = "http://export.arxiv.org/api/query"

    def __init__(self, nodes_path, rels_path, llm_client,
                 pdf_dir="data/raw/auto_ingested", max_papers=5,
                 seen_papers_file="data/processed/seen_arxiv_papers.json"):
        self.nodes_path       = nodes_path
        self.rels_path        = rels_path
        self.llm              = llm_client
        self.pdf_dir          = Path(pdf_dir)
        self.max_papers       = max_papers
        self.seen_papers_file = seen_papers_file
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self._seen = self._load_seen()
        self.nodes_df = pd.read_csv(nodes_path)
        self.rels_df  = pd.read_csv(rels_path)
        logger.info("ArXivIngester ready. KG: %d nodes, %d rels. Seen: %d papers.",
                    len(self.nodes_df), len(self.rels_df), len(self._seen))

    def _load_seen(self):
        if os.path.exists(self.seen_papers_file):
            with open(self.seen_papers_file) as f:
                return set(json.load(f))
        return set()

    def _save_seen(self):
        os.makedirs(os.path.dirname(self.seen_papers_file), exist_ok=True)
        with open(self.seen_papers_file,"w") as f:
            json.dump(list(self._seen), f)

    def search_arxiv(self, query, max_results=5):
        params = {"search_query":f"all:{query}","start":0,
                  "max_results":max_results,"sortBy":"submittedDate","sortOrder":"descending"}
        try:
            resp = requests.get(self.ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("ArXiv API error: %s", exc)
            return []

        papers = []
        for entry in re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL):
            try:
                aid = re.search(r"<id>.*?/abs/([^<]+)</id>", entry)
                ttl = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                abs_ = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                pub  = re.search(r"<published>(.*?)</published>", entry)
                auths = re.findall(r"<name>(.*?)</name>", entry)
                if not all([aid,ttl,abs_,pub]): continue
                a = aid.group(1).strip()
                if a in self._seen: continue
                papers.append(ArXivPaper(
                    arxiv_id=a, title=ttl.group(1).strip().replace("\n"," "),
                    abstract=abs_.group(1).strip().replace("\n"," "),
                    authors=auths or ["Unknown"],
                    pdf_url=f"https://arxiv.org/pdf/{a}.pdf",
                    published=pub.group(1)[:10]))
            except Exception: continue
        logger.info("Found %d new papers for: '%s'", len(papers), query)
        return papers[:max_results]

    def _get_text(self, paper):
        pdf_path = self.pdf_dir / f"{paper.arxiv_id.replace('/','_')}.pdf"
        if not pdf_path.exists():
            try:
                r = requests.get(paper.pdf_url, timeout=60)
                r.raise_for_status()
                pdf_path.write_bytes(r.content)
            except Exception:
                return f"TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}"
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            text = "".join(page.get_text() for i,page in enumerate(doc) if i<8)
            doc.close()
            return text[:6000]
        except Exception:
            return f"TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}"

    def _extract(self, paper, text):
        prompt = EXTRACTION_PROMPT.format(text=text[:4000])
        try:
            response, *_ = self.llm.generate(prompt, max_tokens=1500)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m: return [], []
            data = json.loads(m.group())
            nodes = data.get("nodes",[])
            rels  = data.get("relationships",[])
            nodes.append({"node_id":paper.paper_id,"label":"Paper",
                          "value":None,"unit":None,"rung":None})
            for rel in rels:
                if not rel.get("paper_id"): rel["paper_id"] = paper.paper_id
            logger.info("Extracted %d nodes, %d rels from %s", len(nodes), len(rels), paper.arxiv_id)
            return nodes, rels
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            return [], []

    def _merge(self, nodes, rels, paper):
        existing_ids = set(self.nodes_df["node_id"].tolist())
        existing_rels = set(zip(self.rels_df["source_id"],
                                self.rels_df["target_id"],
                                self.rels_df["relationship_type"]))
        new_nodes, new_rels = [], []
        for n in nodes:
            nid = n.get("node_id","").strip()
            if not nid or nid in existing_ids: continue
            new_nodes.append({"node_id":nid,"label":n.get("label","Unknown"),
                              "rung":n.get("rung"),"class":None,"family":None,
                              "unit":n.get("unit"),"value":n.get("value"),"source":paper.paper_id})
            existing_ids.add(nid)
        for r in rels:
            key = (r.get("source_id",""),r.get("target_id",""),r.get("relationship_type",""))
            if not all(key) or key in existing_rels: continue
            new_rels.append({"paper_id":r.get("paper_id",paper.paper_id),
                             "source_id":r["source_id"],"target_id":r["target_id"],
                             "relationship_type":r["relationship_type"],"condition":None})
            existing_rels.add(key)
        if new_nodes:
            self.nodes_df = pd.concat([self.nodes_df,pd.DataFrame(new_nodes)],ignore_index=True)
        if new_rels:
            self.rels_df  = pd.concat([self.rels_df, pd.DataFrame(new_rels)], ignore_index=True)
        logger.info("Merged +%d nodes, +%d rels from %s", len(new_nodes), len(new_rels), paper.arxiv_id)
        return len(new_nodes), len(new_rels)

    def _save_kg(self):
        self.nodes_df.to_csv(self.nodes_path, index=False)
        self.rels_df.to_csv(self.rels_path, index=False)
        logger.info("KG saved: %d nodes, %d rels", len(self.nodes_df), len(self.rels_df))

    def run(self, keywords=None, save_kg=True):
        keywords = keywords or ARXIV_KEYWORDS
        results, processed = [], 0
        for kw in keywords:
            if processed >= self.max_papers: break
            for paper in self.search_arxiv(kw, max_results=3):
                if processed >= self.max_papers: break
                result = ExtractionResult(paper=paper, nodes=[], relationships=[])
                try:
                    text = self._get_text(paper)
                    nodes, rels = self._extract(paper, text)
                    result.nodes, result.relationships = nodes, rels
                    result.status = "extracted"
                    nn, nr = self._merge(nodes, rels, paper)
                    result.new_node_count, result.new_rel_count = nn, nr
                    result.status = "merged"
                    self._seen.add(paper.arxiv_id)
                except Exception as exc:
                    logger.error("Failed %s: %s", paper.arxiv_id, exc)
                    result.status = "failed"
                results.append(result)
                processed += 1
                time.sleep(1)
        if save_kg and any(r.status=="merged" for r in results):
            self._save_kg()
        self._save_seen()
        return results

    def get_summary(self, results):
        lines = ["="*56,"🔄 AUTO-KGR PAPER INGESTION SUMMARY","="*56]
        for r in results:
            icon = "✅" if r.status=="merged" else "❌"
            lines.append(f"{icon} {r.paper.title[:55]}")
            lines.append(f"   ArXiv: {r.paper.arxiv_id} | +{r.new_node_count} nodes | +{r.new_rel_count} rels | {r.status}")
        total_n = sum(r.new_node_count for r in results)
        total_r = sum(r.new_rel_count  for r in results)
        lines += ["="*56, f"Total: +{total_n} nodes, +{total_r} rels added to KG"]
        return "\n".join(lines)
