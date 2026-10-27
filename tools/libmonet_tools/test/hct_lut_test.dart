// HCT (hue, chroma, tone) of the 32,768 colors the on-device graph can
// produce (5-bit codes; color = code channel x 8 + 4), by libmonet's Hct, as
// the app's rule sees them. Float32 little-endian, [code][hue, chroma, tone].
// The graph's rule_v1 step reads it (scripts/graph_rule.py, build_graph.py).
// Run from the repo root: sh scripts/libmonet_tools.sh hct_lut
// It's a test, not a bin/ script, because libmonet's Hct imports dart:ui.
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:libmonet/colorspaces/hct.dart';

void main() {
  test('hct lut', () {
    final path = Platform.environment['MONET_HCT_LUT'];
    if (path == null) fail('set MONET_HCT_LUT (output path)');
    final out = Float32List(32768 * 3);
    for (var code = 0; code < 32768; code++) {
      final r = (code >> 10) * 8 + 4, g = ((code >> 5) & 31) * 8 + 4;
      final b = (code & 31) * 8 + 4;
      final h = Hct.fromInt(0xff000000 | (r << 16) | (g << 8) | b);
      out[code * 3] = h.hue;
      out[code * 3 + 1] = h.chroma;
      out[code * 3 + 2] = h.tone;
    }
    File(path)
      ..parent.createSync(recursive: true)
      ..writeAsBytesSync(out.buffer.asUint8List());
    // ignore: avoid_print
    print('hct lut: 32768 colors -> $path');
  });
}
