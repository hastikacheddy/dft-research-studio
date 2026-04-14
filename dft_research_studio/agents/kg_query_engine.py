from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

TASK_BENCHMARKS = {
    "Non-covalent interactions":    ["S66","S22","NCIBLIND10","S30L","NCED","SCONF"],
    "Main-group thermochemistry":   ["W4-11","GMTKN30","GMTKN55","AE6","G3"],
    "Reaction barrier heights":     ["BH76","HTBH38","NHTBH38","DBH76","BBH7"],
    "Transition metal complexes":   ["TMC32","3dTMRE18","TMAE9","MOR41","MLBE21"],
    "Dispersion-dominated systems": ["S22","S66","S30L","PCONF21","ICONF"],
    "Excited states":               ["CT7","EE69","CTS8","MAEE5"],
}

PAPER_CITATIONS = {
    "Caldeweyher2019":          "Caldeweyher et al. (2019) J. Chem. Phys. 150, 154122 — D4 dispersion",
    "GoerigkGrimme2011":        "Goerigk & Grimme (2011) PCCP 13, 6670 — GMTKN30",
    "GoerigkGrimme2017":        "Goerigk et al. (2017) PCCP 19, 32184 — GMTKN55",
    "GoerigkGrimme2017_SI":     "Goerigk et al. (2017) SI — GMTKN55 extended data",
    "ZhaoTruhlar2008":          "Zhao & Truhlar (2008) Theor. Chem. Acc. 120, 215 — M06 suite",
    "ZhaoTruhlar2006":          "Zhao & Truhlar (2006) J. Chem. Phys. 125, 194101 — M06-L",
    "ZhaoTruhlar2005":          "Zhao & Truhlar (2005) J. Phys. Chem. A 109, 5656",
    "ZhaoTruhlar2008_M08":      "Zhao & Truhlar (2008) JCTC 4, 1849 — M08",
    "MardirossianHead-Gordon2017":"Mardirossian & Head-Gordon (2017) Mol. Phys. 115, 2315",
    "Mardirossian2017":         "Mardirossian & Head-Gordon (2017) Mol. Phys. 115, 2315",
    "Perdew1996":               "Perdew, Burke & Ernzerhof (1996) PRL 77, 3865 — PBE",
    "PerdewBurkeErnzerhof1996": "Perdew, Burke & Ernzerhof (1996) PRL 77, 3865 — PBE",
    "Becke1993":                "Becke (1993) J. Chem. Phys. 98, 5648 — B3LYP",
    "Becke1988":                "Becke (1988) Phys. Rev. A 38, 3098 — B88 exchange",
    "Grimme2006":               "Grimme (2006) J. Comput. Chem. 27, 1787 — D2 dispersion",
    "Grimme2010":               "Grimme et al. (2010) J. Chem. Phys. 132, 154104 — D3",
    "Grimme2011":               "Grimme et al. (2011) J. Comput. Chem. 32, 1456 — D3(BJ)",
    "GrimmeEhrlichGoerigk2011": "Grimme, Ehrlich & Goerigk (2011) J. Comput. Chem. 32, 1456",
    "Řezáč2011":                "Rezac et al. (2011) JCTC 7, 2427 — S66 dataset",
    "Rezac2011":                "Rezac et al. (2011) JCTC 7, 2427 — S66 dataset",
    "RezacHobza2011":           "Rezac & Hobza (2011) JCTC 7, 2427 — S66",
    "Jurečka2006":              "Jurecka et al. (2006) PCCP 8, 1985 — S22",
    "Jurecka2006":              "Jurecka et al. (2006) PCCP 8, 1985 — S22",
    "Yu2016":                   "Yu et al. (2016) Chem. Sci. 7, 5032 — MN15",
    "YuTruhlar2016":            "Yu & Truhlar (2016) Chem. Sci. 7, 5032 — MN15",
    "ChaiHeadGordon2008":       "Chai & Head-Gordon (2008) PCCP 10, 6615 — wB97X-D",
    "ChainHeadGordon2008":      "Chai & Head-Gordon (2008) PCCP 10, 6615 — wB97X-D",
    "Chai2008":                 "Chai & Head-Gordon (2008) J. Chem. Phys. 128, 084106",
    "Chai2008_Disp":            "Chai & Head-Gordon (2008) PCCP 10, 6615 — dispersion",
    "PeveratiTruhlar2011":      "Peverati & Truhlar (2011) J. Phys. Chem. Lett. 2, 2810",
    "PeveratiTruhlar2014":      "Peverati & Truhlar (2014) Phil. Trans. R. Soc. A 372",
    "Baskerville2022":          "Baskerville et al. (2022) — DFT assessment",
    "Goerigk2011":              "Goerigk & Grimme (2011) PCCP 13, 6670",
    "The ABC of DFT":           "Koch & Holthausen — A Chemist's Guide to DFT",
}


class KGQueryEngine:
    def __init__(self, nodes_df: pd.DataFrame, rels_df: pd.DataFrame) -> None:
        self.nodes = nodes_df
        self.rels  = rels_df
        self._vr_nodes   = nodes_df[nodes_df["label"] == "ValidationResult"].copy()
        self._func_nodes = nodes_df[nodes_df["label"] == "Functional"]["node_id"].tolist()
        self._rfm  = rels_df[rels_df["relationship_type"] == "RESULT_FOR_METHOD"].copy()
        self._rfb  = rels_df[rels_df["relationship_type"] == "RESULT_FOR_BENCHMARK"].copy()
        self._fail = rels_df[rels_df["relationship_type"].str.startswith("FAILS", na=False)].copy()
        logger.info("KGQueryEngine: %d VR nodes, %d functionals, %d failure edges",
                    len(self._vr_nodes), len(self._func_nodes), len(self._fail))

    def _resolve_functional(self, functional: str) -> List[str]:
        clean = functional.strip()
        aliases = {
            "M06-2X":["M062X","M06-2X"],"M062X":["M062X","M06-2X"],
            "M06-L":["M06L","M06-L"],"M06L":["M06L","M06-L"],
            "wB97X-D":["wB97X-D","wB97XD"],
        }
        candidates = [clean] + aliases.get(clean, []) + [clean.replace("-","")]
        return list({c for c in candidates if c in self._func_nodes}) or [clean]

    def get_mae_for_functional(self, functional: str, task: str, max_results: int = 8) -> List[Dict]:
        resolved = self._resolve_functional(functional)
        task_key = next((k for k in TASK_BENCHMARKS if k.lower() in task.lower()), None)
        relevant = set(TASK_BENCHMARKS.get(task_key, []))
        results, seen = [], set()

        for func_id in resolved:
            for vr_id in self._rfm[self._rfm["target_id"]==func_id]["source_id"].tolist():
                if vr_id in seen: continue
                seen.add(vr_id)
                vr_row = self._vr_nodes[self._vr_nodes["node_id"]==vr_id]
                if vr_row.empty: continue
                val  = vr_row.iloc[0]["value"]
                unit = vr_row.iloc[0]["unit"]
                if pd.isna(val) and pd.isna(unit): continue
                actual_val  = str(val)  if pd.notna(val)  else str(unit)
                actual_unit = str(unit) if pd.notna(unit) and str(unit) != actual_val else ""
                bench_row = self._rfb[self._rfb["source_id"]==vr_id]["target_id"]
                benchmark = bench_row.iloc[0] if not bench_row.empty else vr_id
                paper_rows = self.rels[self.rels["source_id"]==vr_id]["paper_id"].dropna()
                paper_id   = paper_rows.iloc[0] if not paper_rows.empty else "KG"
                priority   = 2 if any(rb.upper() in vr_id.upper() for rb in relevant) else 0
                results.append({"functional":functional,"vr_node":vr_id,"benchmark":str(benchmark),
                                 "value":actual_val,"unit":actual_unit,"paper":str(paper_id).strip(),"priority":priority})

        results.sort(key=lambda x:(-x["priority"],len(x["vr_node"])))
        return results[:max_results]

    def get_failure_modes(self, functional: str, task: str, max_results: int = 3) -> List[str]:
        resolved = self._resolve_functional(functional)
        failures, seen = [], set()
        for func_id in resolved:
            for _, row in self._fail[self._fail["source_id"]==func_id].iterrows():
                readable = row["relationship_type"].replace("FAILS_","").replace("FAILS ","").replace("_"," ").strip()
                text = f"{readable} [{row['target_id']}]"
                if text not in seen:
                    seen.add(text)
                    failures.append(text)
        return failures[:max_results]

    def format_citation(self, paper_id: str) -> str:
        if not paper_id or paper_id in ("KG","Unknown","nan"): return ""
        return PAPER_CITATIONS.get(paper_id.strip(), paper_id.strip())

    def build_comparison_data(self, accepted_func, rejected_funcs, task, papers):
        all_funcs = [accepted_func] + rejected_funcs[:2]
        rows = []
        for func in all_funcs:
            mae_data = self.get_mae_for_functional(func, task)
            failures = self.get_failure_modes(func, task)
            task_mae = [m for m in mae_data if m["priority"]==2]
            best_mae = task_mae[0] if task_mae else (mae_data[0] if mae_data else None)
            rows.append({"functional":func,"accepted":(func==accepted_func),
                          "mae_value":best_mae["value"] if best_mae else "—",
                          "mae_unit":best_mae["unit"] if best_mae else "",
                          "benchmark":best_mae["benchmark"] if best_mae else "—",
                          "paper":best_mae["paper"] if best_mae else "",
                          "failures":failures[:2],"all_mae":mae_data})
        citations = [self.format_citation(p) for p in papers if self.format_citation(p)]
        return {"rows":rows,"citations":citations[:4]}
