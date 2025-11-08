from collections import Counter
from itertools import combinations
from typing import Dict, List
from collections import Counter
from typing import Dict, List


def make_segment(role: str | None = None, geo: str | None = None, age: int | None = None) -> str:
    r = role or "user"
    g = geo or "UNK"
    a = "na"
    if isinstance(age, int):
        a = "y" if age < 30 else "m" if age < 60 else "s"
    return f"role:{r}|geo:{g}|age:{a}"

def frequent_pairs(events: List[Dict], min_support: int = 5) -> Dict[str, int]:
    counts: Counter = Counter()
    for e in events:
        
        codes = set(e.get("payload", {}).get("symptom_codes", []))
        for a, b in combinations(sorted(codes), 2):
            counts[(a, b)] += 1
    return {f"{a},{b}": c for (a, b), c in counts.items() if c >= min_support}

def extract_symptom_features(events: List[Dict]) -> Dict:
    c = Counter()
    for e in events:
        c.update(e.get("payload", {}).get("symptom_codes", []))
    return {
        "most_common": c.most_common(5),
        "total_unique": len(c),
        "total_occurrences": sum(c.values()),
    }
