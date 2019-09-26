import collections

__all__ = ['get_early_exit']

EarlyExit = collections.namedtuple('EarlyExit', ['in_channels', 'final_channels', 'thres', 'blocks_idx', 'device'])

EarlyExit.__new__.__defaults__ = (None,) * len(EarlyExit._fields)

def get_early_exit(in_channels=40, final_channels=150, thres=None, blocks_idx=None, device='cpu'):
    early_exit = EarlyExit(
        in_channels=in_channels,
        final_channels=final_channels,
        thres=thres,
        blocks_idx=blocks_idx,
        device=device
    )
    
    return early_exit