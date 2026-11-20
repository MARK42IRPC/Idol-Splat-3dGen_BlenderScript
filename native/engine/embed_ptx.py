"""Embed a PTX text file as a C array header."""
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    data = a.ptx.read_bytes() + b"\0"
    lines = ["// generated from %s" % a.ptx.name]
    lines.append("static const unsigned char kIdolPtx[] = {")
    for i in range(0, len(data), 20):
        lines.append("  " + ",".join(str(b) for b in data[i:i+20]) + ",")
    lines.append("};")
    a.out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("embedded", len(data), "bytes ->", a.out)

if __name__ == "__main__":
    main()
