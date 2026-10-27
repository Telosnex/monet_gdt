# CONCLUDED 2026-10-25. Winners: arm a2 (hex + HCT + share labels), the
# vivid merge, and the rule_v1 triad (beat the on-device rule 27-11). They
# now live in telosnex lib/features/system_agents/wallpaper_color_decisions.dart
# (parity with rule_v1.py: 60/60); these scripts are the record, not
# production. Votes and Jev answers: results/.
#
# 50 new images for the color A/B : 25 megalith,
# 13 generic, 12 genre from the downloaded corpus, none already in
# monet_color_ab. Order: sha256("more50:" + id). Copies to telosnex
# dev/wallpaper_colors/monet_color_ab2/<part>__<id>.jpg.
import hashlib
import os
import shutil

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TX = os.path.join(
    os.environ['TELOSNEX'], 'dev', 'wallpaper_colors'
)  # TELOSNEX: Telosnex app checkout (maintainer-only)
used = {f.split('__')[1].split('.')[0] for f in os.listdir(f'{TX}/monet_color_ab')}
os.makedirs(f'{TX}/monet_color_ab2', exist_ok=True)
n = 0
for part, k in (('megalith', 25), ('generic', 13), ('genre', 12)):
    ids = sorted(
        (f[:-4] for f in os.listdir(f'{R}/data/images/{part}') if f.endswith('.jpg') and f[:-4] not in used),
        key=lambda i: hashlib.sha256(f'more50:{i}'.encode()).hexdigest(),
    )[:k]
    for i in ids:
        shutil.copy(f'{R}/data/images/{part}/{i}.jpg', f'{TX}/monet_color_ab2/{part}__{i}.jpg')
        n += 1
print(n, 'images copied')
