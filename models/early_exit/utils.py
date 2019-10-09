import collections

__all__ = ['get_early_exit']

EarlyExit = collections.namedtuple('EarlyExit', ['in_channels', 'final_channels', 'input_size', 'use_bias', 'thres', 'blocks_idx', 'device'])

EarlyExit.__new__.__defaults__ = (None,) * len(EarlyExit._fields)

def get_early_exit(in_channels=None, final_channels=150, input_size=None, use_bias=False, thres=None, blocks_idx=None, device='cpu'):
    """
    returns namedtuple for counting module.
    """
    early_exit = EarlyExit(
        in_channels=in_channels,
        final_channels=final_channels,
        input_size=input_size,
        use_bias=use_bias,
        thres=thres,
        blocks_idx=blocks_idx,
        device=device
    )
    
    return early_exit