// v2 triads for out/v2/thumb96_<shard>.bin (scripts/v2_prep.py): Celebi
// with MONET_V2_CLUSTERS (default 128) clusters on each 96 px RGBA
// thumbnail, then monet_ui's MonetGdtRule20261030. One jsonl per shard:
// out/v2/labels_<shard>.jsonl, rows {i, triad: [P, S, T] hex}; plus
// out/v2/labels_meta.json (scorer sha256, settings).
// Env: MONET_V2_DIR (out/v2), MONET_V2_SHARDS ("a-b", inclusive),
// MONET_V2_SCORER (path of monet_gdt_rule_2026_10_30.dart, for the sha256).
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:monet_ui/extract/quantizer_celebi.dart';
import 'package:monet_ui/extract/monet_gdt_rule_2026_10_30.dart';

String hex(int a) => '#${(a & 0xffffff).toRadixString(16).padLeft(6, '0')}';

void main() {
  test('v2 labels', () async {
    final env = Platform.environment;
    final dir = env['MONET_V2_DIR'];
    final range = env['MONET_V2_SHARDS'];
    final scorer = env['MONET_V2_SCORER'];
    if (dir == null || range == null || scorer == null) {
      fail('set MONET_V2_DIR, MONET_V2_SHARDS (a-b), MONET_V2_SCORER');
    }
    final clusters = int.parse(
      env['MONET_V2_CLUSTERS'] ?? '$kMonetGdtRule20261030Clusters',
    );
    final sha = sha256.convert(File(scorer).readAsBytesSync()).toString();
    File('$dir/labels_meta.json').writeAsStringSync(
      jsonEncode({
        'scorer': 'monet_ui lib/extract/monet_gdt_rule_2026_10_30.dart',
        'scorer_sha256': sha,
        'quantizer': 'QuantizerCelebi',
        'clusters': clusters,
        'thumbnail': 'telosnex box fold: <=2048, 384, 96 (scripts/v2_prep.py)',
      }),
    );
    final [a, b] = range.split('-').map(int.parse).toList();
    for (var s = a; s <= b; s++) {
      final name = s.toString().padLeft(3, '0');
      final inp = File('$dir/thumb96_$name.bin');
      final out = File('$dir/labels_$name.jsonl');
      if (!inp.existsSync() || out.existsSync()) continue;
      final bytes = inp.readAsBytesSync();
      final bd = ByteData.sublistView(bytes);
      final buf = StringBuffer();
      var off = 0, i = 0;
      while (off < bytes.length) {
        final w = bd.getUint32(off, Endian.little),
            h = bd.getUint32(off + 4, Endian.little);
        off += 8;
        final px = <int>[];
        for (var p = 0; p < w * h; p++) {
          final o = off + p * 4;
          if (bytes[o + 3] != 0xff) continue;
          px.add(
            0xff000000 | (bytes[o] << 16) | (bytes[o + 1] << 8) | bytes[o + 2],
          );
        }
        off += w * h * 4;
        final q = await QuantizerCelebi().quantize(px, clusters);
        final t = MonetGdtRule20261030.score(q, pixels: px).triad;
        buf.writeln(
          jsonEncode({
            'i': i,
            'triad': [for (final o in t) hex(o.argb)],
          }),
        );
        i++;
      }
      File('${out.path}.tmp').writeAsStringSync(buf.toString());
      File('${out.path}.tmp').renameSync(out.path);
    }
  }, timeout: Timeout.none);
}
