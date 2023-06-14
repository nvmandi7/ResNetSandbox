
import torch
from torch import nn

class DebugSelector:
    '''
    Choose from a set of prebuilt debug routines by name
    '''

    debug_routines = {
        'resnet_forward_hooks': DebugSelector.add_resnet_forward_hooks,
        'resnet_backward_hooks': DebugSelector.add_resnet_backward_hooks,
    }

    @classmethod
    def parse_debug(cls, name):
        t_func = cls.debug_routines[name]
        return t_func()
    
    @classmethod
    def add_resnet_forward_hooks(model):
        torch.autograd.set_detect_anomaly(True)

        outputs = {}
        def hook(module, input, output):
            outputs[module.__class__.__name__] = output
            print(output.detach())

        model.stem.register_forward_hook(hook)
        model.block_stack.register_forward_hook(hook)
        model.classification_head.register_forward_hook(hook)
        return outputs

    
    @classmethod
    def add_resnet_backward_hooks():
        pass