"""Saved-result table projection, following V3 compute.numeric field semantics."""
import csv
import math
from pathlib import Path

FIELDS = ['model_id', 'estimate', 'SE', 'CI_low', 'CI_high', 'p_value', 'N', 'status']


def rows(payload):
    result = []
    seen = set()
    for model in payload['models']:
        mid = model['model_id']
        if mid in seen: raise ValueError('DUPLICATE_REPORT_MODEL')
        seen.add(mid)
        if 'coefficients' in model:
            terms = [c for c in model['coefficients'] if c['term'] == model['coefficient_name']]
            if len(terms) != 1: raise ValueError('PRIMARY_SIDECAR_MISSING')
            for field in FIELDS[1:6]:
                key = 'coefficient' if field == 'estimate' else field
                a, b = model[field], terms[0][key]
                if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                         or not math.isfinite(v)) for v in (a, b)):
                    raise ValueError('NONFINITE_OR_INVALID_REPORT_VALUE')
                if (a is None) != (b is None) or (a is not None and abs(a-b) > 1e-10):
                    raise ValueError('NUMERIC_SIDECAR_MISMATCH')
        else:
            raise ValueError('COEFFICIENT_SIDECAR_REQUIRED')
        result.append({key: model.get(key) for key in FIELDS})
    if not result: raise ValueError('EMPTY_RESULTS')
    return result


def write_table(path, records):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(records)
