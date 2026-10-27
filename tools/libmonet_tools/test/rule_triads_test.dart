// rule_v1 triads (libmonet's monetGdtRuleV1Triad, the code the app runs)
// for every row of jev/colors.jsonl -> MONET_TRIADS_OUT (jsonl: id, triad,
// and each option's rounded HCT for the graph parity check,
// scripts/check_graph_rule.py). The color model trains on these as targets.
// Run from the repo root: sh scripts/libmonet_tools.sh rule_triads
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:libmonet/colorspaces/hct.dart';
import 'package:libmonet/extract/monet_gdt_rule_v1.dart';

void main() {
  test('rule triads', () {
    final labels = Platform.environment['MONET_JEV_COLORS'];
    final outPath = Platform.environment['MONET_TRIADS_OUT'];
    if (labels == null || outPath == null) {
      fail('set MONET_JEV_COLORS (jev/colors.jsonl) and MONET_TRIADS_OUT');
    }
    final out = StringBuffer();
    var n = 0, skipped = 0;
    for (final l in File(labels).readAsLinesSync()) {
      final r = jsonDecode(l) as Map<String, dynamic>;
      if (r['error'] != null) continue;
      final options = [
        for (final o in (r['options'] as List).cast<Map>()) o['hex'] as String,
      ];
      final p = r['probabilities'] as Map<String, dynamic>;
      final triad = monetGdtRuleV1Triad(options, [
        for (final role in kWallpaperColorRoles)
          (p[role] as Map).map(
            (k, v) => MapEntry(k as String, (v as num).toDouble()),
          ),
      ]);
      if (triad == null) {
        skipped++;
        continue;
      }
      out.writeln(
        jsonEncode({
          'id': r['id'],
          'triad': triad,
          'hct': [
            for (final h in options)
              () {
                final c = Hct.fromInt(
                  0xff000000 | int.parse(h.substring(1), radix: 16),
                );
                return [c.hue.round(), c.chroma.round(), c.tone.round()];
              }(),
          ],
        }),
      );
      n++;
    }
    File(outPath)
      ..parent.createSync(recursive: true)
      ..writeAsStringSync('$out');
    // ignore: avoid_print
    print('rule triads: $n, skipped $skipped (< 3 options) -> $outPath');
  });
}
