#!/usr/bin/env python3
"""Minimal, careful GGUF metadata reader (v2 and v3). Prints KV metadata only."""
import struct
import sys

# value types
T = {
    "GGUF_TYPE_UINT8": 0, "GGUF_TYPE_INT8": 1, "GGUF_TYPE_UINT16": 2,
    "GGUF_TYPE_INT16": 3, "GGUF_TYPE_UINT32": 4, "GGUF_TYPE_INT32": 5,
    "GGUF_TYPE_FLOAT32": 6, "GGUF_TYPE_BOOL": 7, "GGUF_TYPE_STRING": 8,
    "GGUF_TYPE_ARRAY": 9, "GGUF_TYPE_UINT64": 10, "GGUF_TYPE_INT64": 11,
    "GGUF_TYPE_FLOAT64": 12,
}
NAME = {v: k for k, v in T.items()}

def rd_str(buf, off):
    n = struct.unpack_from("<Q", buf, off)[0]; off += 8
    s = buf[off:off+n].decode("utf-8"); off += n
    return s, off

def rd_val(buf, off, vtype, depth=0):
    if depth > 4:
        raise RuntimeError("array nesting too deep")
    if vtype == 0:  # uint8
        return struct.unpack_from("<B", buf, off)[0], off + 1
    if vtype == 1:  # int8
        return struct.unpack_from("<b", buf, off)[0], off + 1
    if vtype == 2:  # uint16
        return struct.unpack_from("<H", buf, off)[0], off + 2
    if vtype == 3:  # int16
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if vtype == 4:  # uint32
        return struct.unpack_from("<I", buf, off)[0], off + 4
    if vtype == 5:  # int32
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if vtype == 6:  # float32
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if vtype == 7:  # bool
        return struct.unpack_from("<?", buf, off)[0], off + 1
    if vtype == 10:  # uint64
        return struct.unpack_from("<Q", buf, off)[0], off + 8
    if vtype == 11:  # int64
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if vtype == 12:  # float64
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if vtype == 8:  # string
        return rd_str(buf, off)
    if vtype == 9:  # array
        etype = struct.unpack_from("<I", buf, off)[0]; off += 4
        elen = struct.unpack_from("<Q", buf, off)[0]; off += 8
        arr = []
        for _ in range(elen):
            v, off = rd_val(buf, off, etype, depth+1)
            arr.append(v)
        return arr, off
    raise RuntimeError(f"bad value type {vtype}")

def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    magic = buf[0:4]
    if magic != b"GGUF":
        print(f"NOT GGUF magic={magic!r}"); return
    version = struct.unpack_from("<I", buf, 4)[0]
    n_tensors = struct.unpack_from("<Q", buf, 8)[0]
    n_kv = struct.unpack_from("<Q", buf, 16)[0]
    off = 24
    kv = {}
    for _ in range(n_kv):
        key, off = rd_str(buf, off)
        vtype = struct.unpack_from("<I", buf, off)[0]; off += 4
        val, off = rd_val(buf, off, vtype)
        kv[key] = val
    print(f"version={version}  n_tensors={n_tensors}  n_kv={n_kv}  meta_end_off={off}")
    print("---- relevant metadata ----")
    for k in sorted(kv):
        if any(x in k for x in ("arch", "n_layer", "n_embd", "n_head", "head", "sliding", "swa", "rope", "rot", "pos", "block", "ffn", "vocab_size", "context", "ple")):
            print(f"  {k} = {kv[k]!r}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/home/bigattichouse/workspace/gguf/gemma-3-270m-it-Q8_0.gguf")
