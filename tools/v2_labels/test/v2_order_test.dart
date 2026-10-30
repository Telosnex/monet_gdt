// MonetGdtRule20261030's role step, verbatim, on triads in MONET_V2_IN (jsonl
// rows {id, triad: [P, S, T] hex}); writes MONET_V2_OUT rows {id, triad}
// in the order that step gives. A single color stands in for an option
// (colorfulShare 1, so isColorful = monetGdtRule20261030IsColorful).
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:monet_ui/core/argb_srgb_xyz_lab.dart';
import 'package:monet_ui/extract/monet_gdt_rule_2026_10_30.dart';

String hex(int a) => '#${(a & 0xffffff).toRadixString(16).padLeft(6, '0')}';

void main() {
  test('order', () {
    final env = Platform.environment;
    final out = StringBuffer();
    for (final l in File(env['MONET_V2_IN']!).readAsLinesSync()) {
      final r = jsonDecode(l) as Map;
      final triad = [
        for (final h in (r['triad'] as List).cast<String>())
          MonetGdtRule20261030Option(
            0xff000000 | int.parse(h.substring(1), radix: 16),
            1,
            1,
          ),
      ];
      // --- verbatim from MonetGdtRule20261030.score ---
      if (triad.length == 3) {
        final p = triad[0];
        final byHue = triad.every((t) => t.isColorful);
        double step(MonetGdtRule20261030Option o) =>
            byHue ? differenceDegrees(o.hue, p.hue) : p.deltaE(o);
        if (step(triad[1]) > step(triad[2])) {
          triad.insert(1, triad.removeAt(2));
        }
      }
      // ---
      out.writeln(
        jsonEncode({
          'id': r['id'],
          'triad': [for (final o in triad) hex(o.argb)],
        }),
      );
    }
    File(env['MONET_V2_OUT']!).writeAsStringSync(out.toString());
  });
}
