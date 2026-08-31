import json
import pickle
import sys
import numpy as np
pkl_path, bin_path, meta_path = sys.argv[1:4]
obj = pickle.load(open(pkl_path, 'rb'))
params = obj['params']; cfg = obj.get('config', {})
layers = int(cfg.get('num_layers', 27))

def arr(path, expected=None):
    x = params
    for part in path.split('.'):
        x = x[part]
    a = np.asarray(x, dtype=np.float32)
    if expected is not None and tuple(a.shape) != tuple(expected): raise RuntimeError(f'{path}: expected {expected}, got {a.shape}')
    return a

def t2(a): return np.ascontiguousarray(a.T, dtype=np.float32)
weights = [arr('embedding.embedding')]
p='stack.layers.block.'
for l in range(layers):
    weights.extend([
      arr(f'{p}ZCRMSNorm_0.scale')[l], t2(arr(f'{p}self_attn.q_proj.kernel')[l]), t2(arr(f'{p}self_attn.k_proj.kernel')[l]), t2(arr(f'{p}self_attn.v_proj.kernel')[l]),
      arr(f'{p}self_attn.q_norm.scale')[l], arr(f'{p}self_attn.k_norm.scale')[l], t2(arr(f'{p}self_attn.gate_proj.kernel')[l]), t2(arr(f'{p}self_attn.out_proj.kernel')[l]),
      arr(f'{p}post_attn_norm.scale')[l], arr(f'{p}attn_gate')[l:l+1], arr(f'{p}pre_hada_norm.scale')[l], arr(f'{p}hadamard_mlp.d1')[l], arr(f'{p}hadamard_mlp.d2')[l], arr(f'{p}hadamard_mlp.d3')[l]
    ])
weights += [arr('stack.mhc_a_pre'), arr('stack.mhc_a_post'), arr('stack.mhc_a_res'), arr('stack.mhc_b_pre'), arr('stack.mhc_b_post'), arr('stack.mhc_b_res'),
  np.ascontiguousarray(np.transpose(arr('stack.mhc_phi_pre'),(0,2,1)).reshape(108,2048)), np.ascontiguousarray(np.transpose(arr('stack.mhc_phi_post'),(0,2,1)).reshape(108,2048)), np.ascontiguousarray(np.transpose(arr('stack.mhc_phi_res'),(0,2,1)).reshape(432,2048))]
for name in ('engrams_0','engrams_1'):
    weights += [np.ascontiguousarray(arr(f'{name}.embedding').reshape(-1,128)), t2(arr(f'{name}.key_proj.kernel')), t2(arr(f'{name}.value_proj.kernel')), arr(f'{name}.taps')]
weights.append(arr('stack.final_norm.scale'))
with open(bin_path,'wb') as f:
    for a in weights: f.write(np.ascontiguousarray(a,dtype=np.float32).astype('<f4',copy=False).tobytes())
meta={'format_version':int(obj.get('format_version',0)),'num_weights':len(weights),'config':cfg,'shapes':[list(a.shape) for a in weights]}
json.dump(meta,open(meta_path,'w',encoding='utf-8'),ensure_ascii=False)
print(f'exported {len(weights)} weights')
