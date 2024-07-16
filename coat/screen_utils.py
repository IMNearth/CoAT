from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator, Dict
from PIL import Image, ImageDraw, ImageFont
from functools import cmp_to_key
import numpy as np

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"

# ========================================================================================

def node_compare(node1, node2):
    if node1 is None and node2 is None:
        return 0
    if node1 is None:
        return 1
    if node2 is None:
        return -1

    l1, t1, r1, b1 = node1['bounds']  # top_left, right_bot
    l2, t2, r2, b2 = node2['bounds']

    base_height = 0
    if t1 > t2:    # node1的上边沿 低于 node2的上边沿
        base_height = (t2 + b2) / 2
        if t1 > base_height: return 1   # 说明node1的左上角在node2中心以下     --》 node1在node2的下一层
    elif t1 < t2:  # node1的上边沿 高于 node2的上边沿
        base_height = (t1 + b1) / 2
        if t2 > base_height: return -1  # 说明node2的左上角在node1的中心以下   --》 node2在node1的下一层

    width_diff = l1 - l2
    if width_diff > 0: return 1         # node1在node2的右边
    elif width_diff < 0: return -1

    return 0


def row_col_sort(nodes):
    """ 节点排序：从左到右, 从上到下

    从上到下先判断元素是否在一行内, 只对同一行的元素从左往右进行排序
    """
    if len(nodes) <= 1: return nodes

    # 首先按照y轴进行排序, 需要同时考虑 y 轴的最小值和中心点的 y 轴坐标
    # nodes.sort(key=lambda x: (x['bounds'][1], (x['bounds'][1] + x['bounds'][3])*0.5))
    
    # 更新排序规则 -- 参考OCR
    nodes.sort(key=cmp_to_key(node_compare))

    first_node = nodes[0]
    other_node = nodes[1:]
    sort_dots = [[first_node]]

    line_index = 0
    for node in other_node:
        center_node_y = (node['bounds'][1] + node['bounds'][3]) * 0.5

        line_nodes = sort_dots[line_index]
        prev_avg_center_y = sum([(x['bounds'][1] + x['bounds'][3])*0.5 for x in line_nodes]) / len(line_nodes)

        if (node['bounds'][1] < prev_avg_center_y < node['bounds'][3]) \
            and (line_nodes[-1]['bounds'][1] < center_node_y < line_nodes[-1]['bounds'][3]):
            #  当前行的平均 y 中心点位于当前结点的 y 轴范围内, 并且
            #  Y轴中心大于上一个点Y轴最小值、小于上一个点Y轴最大值 => 说明在同一行
            sort_dots[line_index].append(node)
        else: # 第二行或新增一行
            line_index += 1
            sort_dots.append([node])

    for dot in sort_dots:  # 对每一行做X轴最小点排序
        dot.sort(key=lambda x: x['bounds'][0])
    
    new_nodes = [dot for dots in sort_dots for dot in dots]
    return new_nodes

# ========================================================================================


def draw_bbox(image_path, bboxes:List[Tuple[float, float, float, float]], 
              texts=None, rgba_color=(0, 0, 255, 0), thickness=1, ret_corrds=False):
    """ Draw the bounding boxes with corresponding texts """
    image = Image.open(image_path)
    w, h = image.size

    text_coords = []

    with image.convert('RGBA') as base:
        tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
        # get a drawing context
        draw = ImageDraw.Draw(tmp)
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox[:4]
            xmin = min(max(0, xmin), w)
            xmax = min(max(xmin, xmax), w)
            ymin = min(max(0, ymin), h)
            ymax = min(max(ymin, ymax), h)
            # draw the boudning box
            draw.rectangle((xmin, ymin, xmax, ymax), outline=rgba_color, width=thickness)
        
            # draw text if any
            if texts:
                text = texts[idx]
                box_height, box_width = ymax - ymin, xmax - xmin
                font_size = int(min(max(int(box_height * 0.7), 14), 36))

                font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")
                left, top, right, bot = font.getbbox(text)
                coords = [
                    xmin, ymin,
                    xmin + right*1.1, ymin,
                    xmin + right*1.1, ymin - bot*1.1,
                    xmin, ymin - bot*1.1
                ]
                draw.polygon(coords, fill=rgba_color)
                draw.text((xmin, ymin - bot*1.05), text, (255,255,255,255), font=font)
                text_coords.append([coords[0], coords[5], coords[2], coords[1]])
        
        out = Image.alpha_composite(base, tmp)

        if ret_corrds: return out, text_coords
        return out


def enlarge_bbox(bbox_list, scale_factor=1.2)->np.ndarray:
    """
    将每个 bounding box 放大一定倍数。

    :param bbox_list: bounding box 列表, 每个 bbox 是一个包含四个值的元组或列表, 表示 (xmin, ymin, xmax, ymax)
    :param scale_factor: 放大倍数
    :return: 放大后的 bounding box 列表
    """
    bbox_array = np.array(bbox_list)
    x_min, y_min, x_max, y_max = \
        bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    
    # 计算每个 bounding box 的中心点
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # 计算每个 bounding box 的宽度和高度
    width = (x_max - x_min) * scale_factor
    height = (y_max - y_min) * scale_factor
    
    # 计算放大后的 bounding box 的新的坐标
    new_x_min = x_center - width / 2
    new_y_min = y_center - height / 2
    new_x_max = x_center + width / 2
    new_y_max = y_center + height / 2
    
    # 将新的坐标组合成 bounding box 列表
    enlarged_bbox_list = np.vstack((new_x_min, new_y_min, new_x_max, new_y_max)).T
    
    return enlarged_bbox_list



def check_inside(x, y, bbox_array):
    """
    判断一个坐标 (x, y) 是否在一个 bounding box 列表里面, 使用 NumPy 以提高效率。
    同时返回所在的所有 bounding box 的坐标。

    :param x: 坐标的 x 值
    :param y: 坐标的 y 值
    :param bbox_array: bounding box 列表, 每个 bbox 是一个包含四个值的元组或列表, 表示 (xmin, ymin, xmax, ymax)
    :return: 一个元组, 第一个元素是布尔值, 如果坐标在任意一个 bounding box 内为 True, 否则为 False；
             第二个元素是包含所有所在 bounding box 坐标的列表
    """
    x_min, y_min, x_max, y_max = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    
    # 检查 (x, y) 是否在任意一个 bounding box 内
    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y
    
    if np.any(within_bbox):
        within_bbox_coords = bbox_array[within_bbox]
        return True, within_bbox_coords
    else:
        return False, None


def intersect_iou(can_bbox, ref_bboxes):
    """
    计算一个边界框和一组边界框的IoU。
    
    参数:
    - can_bbox: NumPy数组或列表list, 形状为[4,], 表示一个边界框[x_min, y_min, x_max, y_max]
    - ref_bboxes: NumPy数组, 形状为[N, 4], 表示N个边界框
    
    返回:
    - ious: NumPy数组, 形状为[N,], 表示输入边界框和每个边界框的IoU
    """
    # 计算交集的坐标
    inter_xmin = np.maximum(can_bbox[0], ref_bboxes[:, 0])
    inter_ymin = np.maximum(can_bbox[1], ref_bboxes[:, 1])
    inter_xmax = np.minimum(can_bbox[2], ref_bboxes[:, 2])
    inter_ymax = np.minimum(can_bbox[3], ref_bboxes[:, 3])

    # 计算交集的面积
    inter_area = np.maximum(0, inter_xmax - inter_xmin) * \
                 np.maximum(0, inter_ymax - inter_ymin)
    
    # 计算候选边界框的面积
    can_bbox_area = np.maximum((can_bbox[2] - can_bbox[0]) * \
                               (can_bbox[3] - can_bbox[1]), 1)
    # 计算参考边界框的面积
    ref_bboxes_area = np.maximum((ref_bboxes[:, 2] - ref_bboxes[:, 0]) * \
                                 (ref_bboxes[:, 3] - ref_bboxes[:, 1]), 1)
    
    # 计算并集的面积
    union_area = can_bbox_area + ref_bboxes_area - inter_area
    
    # 计算IoU
    ious = inter_area / union_area
    return ious