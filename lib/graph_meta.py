"""The released graph's ONNX metadata (model-level fields and metadata_props).
Metadata only: applying it never changes the graph's outputs."""

import onnx
from mood_words import WORDS

# Release date, Y.M.D without zero padding (valid semver). A second release
# on the same day adds .N (2026.10.30.1). The first release, 2026.10.27,
# carries 1.0.0 in its metadata. See CHANGELOG.md.
VERSION = '2026.10.30'


def model_version(v=VERSION):
    """ONNX model_version (int64): YYYYMMDD * 100 + N, unique and increasing per release."""
    p = [int(x) for x in v.split('.')]
    assert len(p) in (3, 4) and 1 <= p[1] <= 12 and 1 <= p[2] <= 31, v
    assert v == '.'.join(str(x) for x in p), f'no zero padding: {v}'
    return (p[0] * 10000 + p[1] * 100 + p[2]) * 100 + (p[3] if len(p) == 4 else 0)


DOC = (
    'monet_gdt: from one wallpaper image, 10 mood scores, a 3-color theme and 29 shape scores. '
    'Input `image` uint8 [h, w, 3 or 4]. Outputs: `scores` float [1, 10], `feature` float [1, 192], '
    '`colors` int64 [3, 3] RGB, `shapes` float [1, 29].'
)


def apply(m, shapes):
    """Set the model fields and metadata on onnx ModelProto `m` (in place)."""
    m.producer_name = 'monet_gdt'
    m.producer_version = VERSION
    m.model_version = model_version()
    m.domain = 'com.telosnex.monet_gdt'
    m.doc_string = DOC
    meta = {
        'name': 'monet_gdt',
        'version': VERSION,
        'license': 'MIT (code); Apache-2.0 (PE-Core weights). See NOTICE.',
        'monet_gdt.mood_words': ','.join(WORDS),
        'monet_gdt.shapes': ','.join(shapes),
    }
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    return m


if __name__ == '__main__':  # re-apply to an existing graph: graph_meta.py <in.onnx> <out.onnx>
    import sys

    m = onnx.load(sys.argv[1])
    old = {p.key: p.value for p in m.metadata_props}
    shapes = (old.get('monet_gdt.shapes') or old['libmonet_gdt.shapes']).split(',')
    onnx.save(apply(m, shapes), sys.argv[2])
