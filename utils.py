import numpy as np
def cvt_mm2inch(mm):
    return mm / 25.4

def cvt_inch2mm(inch):
    return inch * 25.4

def cvt_relative2absolute(pos,size,is_array=True):
    if is_array:
        return (pos*size).astype(int)
    else:
        return int(pos*size)

def cvt_absolute2relative(pos,size,is_int_array=False):
    if is_int_array:
        pos=pos.astype(float)
    return pos/size