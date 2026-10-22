"""Separate OCR text confidence from spatial repeatability; no answer strings."""
from collections import defaultdict
from copy import deepcopy
import math
from statistics import median


def easyocr_content_columns(columns, image_shape, image_height=64):
    """Match EasyOCR AlignCollate/NormalizePAD for the single-image caller."""
    height, width = image_shape[:2]
    if min(height, width, columns, image_height) <= 0:
        raise ValueError('Positive image dimensions and CTC length required')
    canvas_width = math.ceil(width / height) * image_height
    resized_width = min(canvas_width, math.ceil(image_height * width / height))
    return columns * resized_width / canvas_width


def family(variant):
    # Stroke/scale variants of one reader are correlated, not independent votes.
    for prefix in ('easy_word', 'paddle_word', 'vector_', 'ink_', 'native_'):
        if variant.startswith(prefix):
            return prefix.rstrip('_')
    return variant


def spatial_consensus(proposals, cap):
    """Median within each input family, then between families; flag instability."""
    if cap <= 0:
        raise ValueError('Positive character height required')
    groups = []
    for proposal in sorted(proposals, key=lambda p: (p['accepted'], p['box_pdf'][0])):
        box = proposal['box_pdf']
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid OCR box')
        def compatible(group):
            if group[0]['accepted'] != proposal['accepted']:
                return False
            # Complete-link grouping prevents chains joining repeated nearby words.
            for old in group:
                other = old['box_pdf']
                overlap = max(0, min(box[2], other[2]) - max(box[0], other[0]))
                if overlap / min(box[2]-box[0], other[2]-other[0]) <= .5:
                    return False
            return True
        group = next((g for g in groups if compatible(g)), None)
        if group is None:
            groups.append([proposal])
        else:
            group.append(proposal)
    results = []
    for group in groups:
        by_family = defaultdict(dict)
        for p in group:
            # Duplicating an observation must not change the estimate.
            by_family[family(p['variant'])][(p['variant'], tuple(p['box_pdf']))] = p
        medians = {name: [median(p['box_pdf'][i] for p in values.values())
                          for i in range(4)] for name, values in by_family.items()}
        center = [median(b[i] for b in medians.values()) for i in range(4)]
        inliers = {name: box for name, box in medians.items()
                   if max(abs(box[i]-center[i]) for i in (0, 2)) <= cap*.20}
        stable = len(inliers) >= 2 and len(inliers)*2 > len(medians)
        if stable:
            center = [median(b[i] for b in inliers.values()) for i in range(4)]
        best = deepcopy(max(group, key=lambda p: p['score']))
        best.update(box_pdf=center, boundary_stable=stable,
                    boundary_method='family_median',
                    boundary_families=medians, boundary_inliers=list(inliers),
                    variants=sorted({p['variant'] for p in group}),
                    boundary_spread_cap=max(max(b[i] for b in medians.values())-
                                            min(b[i] for b in medians.values())
                                            for i in (0, 2))/cap)
        results.append(best)
    return results
