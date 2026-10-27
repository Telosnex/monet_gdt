# Inputs that are not in this repo, for the maintainer-only comparisons
# (D8 held-out set, old models, A/B pages). The build (rebuild_graph.sh and
# the D5 steps) does not need them.
#   LIBMONET_GDT  checkout of libmonet_gdt (the old research repo: old
#                 teacher/color model, D8 Unsplash/Wallhaven labels; not public)
#   TELOSNEX      checkout of the Telosnex app (A/B pages, app-built color
#                 requests, shipped graph)
import os
import sys


def _need(var, what):
    v = os.environ.get(var)
    if not v:
        sys.exit(f'{os.path.basename(sys.argv[0])}: set {var} ({what}); this script is maintainer-only')
    return os.path.expanduser(v)


def libmonet_gdt(*parts):
    return os.path.join(_need('LIBMONET_GDT', 'libmonet_gdt checkout'), *parts)


def telosnex(*parts):
    return os.path.join(_need('TELOSNEX', 'Telosnex app checkout'), *parts)
