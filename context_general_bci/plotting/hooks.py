# Transformer hook
# https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
# Patch to get attention weights
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

r"""
    Usage:
    save_output = SaveOutput()
    patch_attention(transformer.layers[-1].self_attn)
    hook_handle = transformer.layers[-1].self_attn.register_forward_hook(save_output)
"""