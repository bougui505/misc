"""Zero-dependency pure-Python QR code generator for terminal display (ECC-L, V1-V4)."""

from __future__ import annotations

# (data_codewords, ec_codewords, alignment_center_or_None)
_VERSIONS = {
    1: (19, 7, None),
    2: (34, 10, 18),
    3: (55, 15, 22),
    4: (80, 20, 26),
}

# ECC Level L, Mask 0: 15-bit format string
_FORMAT_BITS = 0b111011111000100

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator(degree: int) -> list[int]:
    poly = [1]
    for i in range(degree):
        root = _GF_EXP[i]
        nxt = [0] * (len(poly) + 1)
        for j, coef in enumerate(poly):
            nxt[j] ^= coef
            nxt[j + 1] ^= _gf_mul(coef, root)
        poly = nxt
    return poly


def _rs_encode(data: list[int], ec_len: int) -> list[int]:
    gen = _rs_generator(ec_len)
    buf = data + [0] * ec_len
    for i in range(len(data)):
        coef = buf[i]
        if coef != 0:
            for j in range(1, len(gen)):
                buf[i + j] ^= _gf_mul(gen[j], coef)
    return buf[len(data) :]


def _encode_bytes(payload: bytes) -> tuple[int, list[int]]:
    n = len(payload)
    for ver, (data_cw, ec_cw, _) in _VERSIONS.items():
        if n + 2 <= data_cw:
            bits: list[int] = []
            # Byte mode indicator (0100) + 8-bit length
            for v, width in ((0b0100, 4), (n, 8)):
                for shift in range(width - 1, -1, -1):
                    bits.append((v >> shift) & 1)
            for b in payload:
                for shift in range(7, -1, -1):
                    bits.append((b >> shift) & 1)
            # Terminator (up to 4 zeros)
            rem = data_cw * 8 - len(bits)
            bits.extend([0] * min(4, rem))
            while len(bits) % 8 != 0:
                bits.append(0)
            cw = [
                sum(bits[i + k] << (7 - k) for k in range(8))
                for i in range(0, len(bits), 8)
            ]
            pad = (0xEC, 0x11)
            idx = 0
            while len(cw) < data_cw:
                cw.append(pad[idx % 2])
                idx += 1
            ec = _rs_encode(cw, ec_cw)
            return ver, cw + ec
    raise ValueError("URL too long for QR V1-V4")


def _build_matrix(text: str) -> list[list[bool]]:
    ver, codewords = _encode_bytes(text.encode("utf-8"))
    size = 17 + 4 * ver
    grid: list[list[bool | None]] = [[None] * size for _ in range(size)]
    reserved = [[False] * size for _ in range(size)]

    def set_mod(r: int, c: int, val: bool) -> None:
        if 0 <= r < size and 0 <= c < size:
            grid[r][c] = val
            reserved[r][c] = True

    def place_finder(r0: int, c0: int) -> None:
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < size and 0 <= c < size:
                    in_outer = 0 <= dr <= 6 and 0 <= dc <= 6
                    on_border = dr in (0, 6) or dc in (0, 6)
                    in_inner = 2 <= dr <= 4 and 2 <= dc <= 4
                    set_mod(r, c, in_outer and (on_border or in_inner))

    place_finder(0, 0)
    place_finder(0, size - 7)
    place_finder(size - 7, 0)

    align_c = _VERSIONS[ver][2]
    if align_c is not None:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                val = abs(dr) == 2 or abs(dc) == 2 or (dr == 0 and dc == 0)
                set_mod(align_c + dr, align_c + dc, val)

    for i in range(8, size - 8):
        set_mod(6, i, i % 2 == 0)
        set_mod(i, 6, i % 2 == 0)

    set_mod(4 * ver + 9, 8, True)

    # Reserve format info areas
    for i in range(9):
        reserved[8][i] = True
        reserved[i][8] = True
    for i in range(8):
        reserved[8][size - 1 - i] = True
        reserved[size - 1 - i][8] = True

    # Place data + EC bits
    all_bits = [(cw >> s) & 1 for cw in codewords for s in range(7, -1, -1)]
    bit_idx = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for dc in (0, -1):
                c = col + dc
                if not reserved[r][c]:
                    bit = bool(all_bits[bit_idx]) if bit_idx < len(all_bits) else False
                    bit_idx += 1
                    # Mask 0: (r + c) % 2 == 0
                    grid[r][c] = bit ^ ((r + c) % 2 == 0)
        upward = not upward
        col -= 2

    # Write format info (MSB at bit 14 -> LSB at bit 0)
    fmt = [bool((_FORMAT_BITS >> i) & 1) for i in range(14, -1, -1)]
    coords1 = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8), (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)]
    coords2 = [(size - 1 - i, 8) for i in range(7)] + [(8, size - 8 + i) for i in range(8)]
    for (r, c), b in zip(coords1, fmt):
        grid[r][c] = b
    for (r, c), b in zip(coords2, fmt):
        grid[r][c] = b

    return [[bool(cell) for cell in row] for row in grid]


def render_qr_terminal(text: str) -> str:
    """Render a QR code as Unicode half-blocks (white background for dark terminals)."""
    mat = _build_matrix(text)
    quiet = 2
    size = len(mat) + 2 * quiet
    padded = [[False] * size for _ in range(size)]
    for r, row in enumerate(mat):
        for c, val in enumerate(row):
            padded[r + quiet][c + quiet] = val

    lines: list[str] = []
    for r in range(0, size, 2):
        row_chars: list[str] = []
        for c in range(size):
            top_dark = padded[r][c]
            bot_dark = padded[r + 1][c] if r + 1 < size else False
            # Light = block, Dark = space (so dark terminals show dark modules)
            if not top_dark and not bot_dark:
                row_chars.append("█")
            elif not top_dark and bot_dark:
                row_chars.append("▀")
            elif top_dark and not bot_dark:
                row_chars.append("▄")
            else:
                row_chars.append(" ")
        lines.append("".join(row_chars))
    return "\n".join(lines)
