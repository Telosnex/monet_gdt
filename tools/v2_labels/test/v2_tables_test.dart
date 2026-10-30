// Per-color tables for the graph's role step, from monet_ui (the code the
// app runs), for each of the graph's 32,768 colors (code c: RGB = 8 *
// (c >> 10, (c >> 5) & 31, c & 31) + 4):
//   <dir>/v2_hue.f32       HCT hue (ColorModel.kDefault)
//   <dir>/v2_colorful.u8   monetGdtRule20261030IsColorful
//   <dir>/v2_lab.f32       labFromArgb (L, a, b)
// Env: MONET_V2_DIR.
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:monet_ui/colorspaces/hct.dart';
import 'package:monet_ui/core/argb_srgb_xyz_lab.dart';
import 'package:monet_ui/extract/monet_gdt_rule_2026_10_30.dart';

int codeArgb(int c) =>
    0xff000000 |
    (((c >> 10) * 8 + 4) << 16) |
    ((((c >> 5) & 31) * 8 + 4) << 8) |
    ((c & 31) * 8 + 4);

void main() {
  test('tables', () {
    final dir =
        Platform.environment['MONET_V2_DIR'] ?? fail('set MONET_V2_DIR');
    final hue = Float32List(32768), lab = Float32List(32768 * 3);
    final col = Uint8List(32768);
    for (var c = 0; c < 32768; c++) {
      final a = codeArgb(c);
      final h = Hct.fromInt(a);
      hue[c] = h.hue;
      col[c] = monetGdtRule20261030IsColorful(h) ? 1 : 0;
      final l = labFromArgb(a);
      lab.setAll(c * 3, l);
    }
    File('$dir/v2_hue.f32').writeAsBytesSync(hue.buffer.asUint8List());
    File('$dir/v2_colorful.u8').writeAsBytesSync(col);
    File('$dir/v2_lab.f32').writeAsBytesSync(lab.buffer.asUint8List());
  });
}
