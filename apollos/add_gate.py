# Adds Ouro's early-exit gate (model.early_exit_gate, Linear(hidden, 1)) to an existing Ouro GGUF as cls.output.{weight,bias}.
import sys, json, struct, numpy as np
sys.path.insert(0, 'llama.cpp/gguf-py')
import gguf
src, st, dst = sys.argv[1:4]
with open(st, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    def get(name):
        m = hdr[name]; a, b = m['data_offsets']; f.seek(base + a); raw = f.read(b - a)
        assert m['dtype'] == 'BF16', m['dtype']
        u = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        return u.view(np.float32).reshape(m['shape'])
    gk = [k for k in hdr if 'early_exit_gate' in k]; print(gk, [hdr[k]['shape'] for k in gk])
    w = get([k for k in gk if k.endswith('weight')][0]); b = get([k for k in gk if k.endswith('bias')][0])
print('gate |w| =', float(np.linalg.norm(w)), 'bias =', b)
r = gguf.GGUFReader(src); arch = 'ouro'
wtr = gguf.GGUFWriter(dst, arch)
for k, fld in r.fields.items():
    if k.startswith('GGUF.') or k == 'general.architecture': continue
    t = fld.types[0]; sub = fld.types[-1] if t == gguf.GGUFValueType.ARRAY else None
    wtr.add_key_value(k, fld.contents(), t, sub_type=sub)
for t in r.tensors:
    wtr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
gw = np.ascontiguousarray(w.reshape(1, -1).astype(np.float32)); gb = b.astype(np.float32).reshape(1)
wtr.add_tensor_info('cls.output.weight', gw.shape, gw.dtype, gw.nbytes, gguf.GGMLQuantizationType.F32)
wtr.add_tensor_info('cls.output.bias', gb.shape, gb.dtype, gb.nbytes, gguf.GGMLQuantizationType.F32)
wtr.write_header_to_file(); wtr.write_kv_data_to_file(); wtr.write_ti_data_to_file()
for t in r.tensors: wtr.write_tensor_data(t.data)
wtr.write_tensor_data(gw); wtr.write_tensor_data(gb); wtr.close(); print('wrote', dst)
