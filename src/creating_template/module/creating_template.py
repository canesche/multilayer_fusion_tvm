'''
AN: AnnotationStep
FU: FuseStep
PR: PragmaStep
RE: ReorderStep

FSP: FollowSplitStep
FFSP: FollowFusedSplitStep
SA: StorageAlignStep
CA: ComputeAtStep
CI: ComputeInlineStep
CR: ComputeRootStep
CHR: CacheReadStep
CHW: CacheWriteStep
RF: RfactorStep
'''

## example input

'''
    ([0.023814, 0.0269994, 0.032609], [[], 
    [['CHW', 2, 'local'], ['SP', 2, 0, 1000, [20, 1, 2], 1], ['SP', 2, 4, 700, [1, 700, 1], 1], ['SP', 2, 8, 800, [5], 1], ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], ['FSP', 3, 0, 1, 2], ['FSP', 3, 3, 2, 2], ['RE', 3, [0, 3, 1, 4, 2, 5]], ['CA', 2, 3, 3], ['FU', 3, [0, 1, 2]], ['AN', 3, 0, 3], ['PR', 2, 0, 'auto_unroll_max_step$512'], ['AN', 2, 9, 2]]])
'''

import tvm
from tvm import autotvm

class Template_ansor():

    cfg = autotvm.get_config()
    sch = None
    tensors = [None]

    def __init__(self, s, t) -> None:
        sch = s
        tensors = t

    def space(self, type, values):

        if type == "SP":
            self.SP(values)
        else:
            raise("Not implemented space search")

    def SP(self, values):
        '''
            SP: SplitStep
            ("SP", stage_id, iter_id, loop_extent, lengths, inner_to_outer)
        '''
        assert len(values) == 5
        
        stage_id = values[0]
        iter_id = values[1]
        loop_extent = values[2]
        lengths = values[3]
        inner_to_outer = values[4]

        xo, xi = self.sch[B].split(B.op.axis[0], factor=32)

