"""Pinned official model/server, with request-scoped RNG and strict load audit."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--official-code', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--port', type=int, default=10097)
a = p.parse_args()
sys.path.insert(0, str(a.official_code.resolve()))
logging.basicConfig(level=logging.INFO, force=True)
torch.set_num_threads(3)
torch.cuda.set_per_process_memory_fraction(.60)
torch.manual_seed(20260925)
from deployment.model_server.server_policy import load_policy_from_checkpoint, build_policy_server_metadata
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

expected = 'a52031302c6dc5b813982227255add8d2acb839149a4b90908b179a8f66adbeb'
h = hashlib.sha256()
with a.checkpoint.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2), b''): h.update(block)
assert h.hexdigest() == expected, 'Official checkpoint hash mismatch'
native_load = torch.nn.Module.load_state_dict
loads = []
def checked_load(self, state_dict, *args, **kwargs):
    result = native_load(self, state_dict, *args, **kwargs)
    if any(k.startswith('policy_backend.') for k in state_dict):
        current = self.state_dict()
        def storage_key(tensor):
            return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        loaded_storage = {storage_key(current[k]): k for k in state_dict if k in current}
        aliases = {k: loaded_storage.get(storage_key(current[k])) for k in result.missing_keys}
        genuine_missing = [k for k,v in aliases.items() if v is None]
        assert not genuine_missing and not result.unexpected_keys, (genuine_missing,result.unexpected_keys)
        # The official framework registers shared VLM/flow modules under aliases.
        # Verify storage identity AND every canonical tensor's loaded value.
        unequal = [k for k,v in state_dict.items() if not torch.equal(current[k],v.to(current[k].dtype))]
        assert not unequal, unequal
        loads.append(dict(keys=len(state_dict), missing_aliases=aliases, genuine_missing=genuine_missing,
                          unexpected=result.unexpected_keys, all_canonical_tensors_equal=True))
    return result
torch.nn.Module.load_state_dict = checked_load
policy = load_policy_from_checkpoint(str(a.checkpoint), use_bf16=True)
torch.nn.Module.load_state_dict = native_load
assert loads, 'Did not observe full model checkpoint loading'
metadata = build_policy_server_metadata(policy, ckpt_path=a.checkpoint, server_type='phase2_official_seeded',
                                       env='robotwin', supported_eval_envs=['robotwin'])
a.output.write_text(json.dumps(dict(checkpoint_sha256=h.hexdigest(), loads=loads, metadata=metadata,
                                  dtype='bfloat16', flow_seed='request-specific'), indent=2))
class SeededPolicy:
    def predict_action(self, flow_seed, **kwargs):
        torch.manual_seed(int(flow_seed))
        np.random.seed(int(flow_seed) % 2**32)
        random.seed(int(flow_seed))
        return policy.predict_action(**kwargs)
WebsocketPolicyServer(SeededPolicy(), host='127.0.0.1', port=a.port, metadata=metadata,
                      idle_timeout=3600).serve_forever()
