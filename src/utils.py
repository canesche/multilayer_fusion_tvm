from tvm import autotvm
from scipy import stats
import numpy as np

def print_var_name(variable):
 for name in globals():
     if eval(name) == variable:
        return name

def auto_fusion_schedule(s, cfg, tensors):
    for t in range(0,len(tensors)-1):
        cfg = autotvm.get_config()
        name = "fuse_%d" % t

        size_schedule = 8 if t == 0 or t == len(tensors)-1 else 9

        cfg.define_knob(name, [i for i in range(size_schedule)])

        assert t < len(tensors)

        actual_tensor = tensors[t]
        next_tensor = tensors[t+1]

        if cfg[name].val == 1:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[1])
        elif cfg[name].val == 2:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[2])
        elif cfg[name].val == 3:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[3])
        elif cfg[name].val == 4:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[1])
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[2])
        elif cfg[name].val == 5:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[1])
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[3])
        elif cfg[name].val == 6:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[2])
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[3])
        elif cfg[name].val == 7:
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[1])
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[2])
            s[actual_tensor].compute_at(s[next_tensor], next_tensor.op.axis[3])
        elif size_schedule == 9 and cfg[name].val == 8 and cfg["fuse_%d" % t-1].val == 0:
            s[actual_tensor].compute_inline()

def limited_interval(max_value, interval):
    new_interval = []
    for elem in interval:
        if max_value <= elem:
            continue
        new_interval.append(elem)
    return new_interval


def auto_tile_schedule(s, cfg, tensors, search_space):
    for t in range(0,len(tensors)):
        actual_tensor = tensors[t]
        name_tensor = str(t)
        
        axis = s[actual_tensor].op.axis
        reduce_axis = s[actual_tensor].op.reduce_axis
        
        for i in range(len(axis)):
            name_axis = axis[i].var
            max_value = axis[i].dom.extent.value
            if max_value != 1:
                name_opt = "tile_%s_%s" %(name_axis, name_tensor)
                cfg.define_knob(name_opt, limited_interval(max_value, search_space))
                if cfg[name_opt].val != 0:
                    _, _ = s[actual_tensor].split(axis[i], cfg[name_opt].val)
        
        for i in range(len(reduce_axis)):
            name_axis = reduce_axis[i].var
            max_value = reduce_axis[i].dom.extent.value
            if max_value != 1:
                name_opt = "tile_%s_%s" %(name_axis, name_tensor)
                cfg.define_knob(name_opt, limited_interval(max_value, search_space))
                if cfg[name_opt].val != 0:
                    _, _ = s[actual_tensor].split(reduce_axis[i], cfg[name_opt].val)

def get_best_time(log, ms=True):
    import json

    f = open(log, "r")
    best_avg = 9999.0
    best_cfg = {}
    for line in f.readlines():
        data = json.loads(line)
        r = np.mean(data["result"][0])
        if (np.mean(best_avg) > r):
            best_avg = data["result"][0]
            best_cfg = data["config"]["entity"]
    f.close()

    if ms: # convet to ms
        best_avg *= 1000
    return best_avg, best_cfg


def p_value(elem_1, elem_2):
    return stats.ttest_ind(elem_1, elem_2).pvalue