import numpy as np


def cvt_mm2inch(mm):
    return mm / 25.4


def cvt_inch2mm(inch):
    return inch * 25.4


def cvt_relative2absolute(pos, size, is_array=True):
    if is_array:
        return (pos*size).astype(int)
    else:
        return int(pos*size)


def cvt_absolute2relative(pos, size, is_int_array=False):
    if is_int_array:
        pos = pos.astype(float)
    return pos/size


def adjast_range(pos, size):
    left, top, right, bottom = pos
    image_size_x, image_size_y = size
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    width = right - left
    height = bottom - top
    if left < 0:
        new_width = center_x * 2
        height = height * new_width / width
        width = new_width
    if top < 0:
        new_height = center_y * 2
        width = width * new_height / height
        height = new_height
    if right >= image_size_x:
        new_width = (image_size_x - center_x)
        height = height * new_width / width
        width = new_width
    if bottom >= image_size_y:
        new_height = (image_size_y - center_y)
        width = width * new_height / height
        height = new_height

    left = center_x - width * 0.5
    top = center_y - height * 0.5
    right = center_x + width * 0.5
    bottom = center_y + height * 0.5

    return [left, top, right, bottom]
