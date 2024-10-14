import os
import re
import json
import imagesize
import numpy as np
import pandas as pd
import Levenshtein
from collections import defaultdict
from .screen_utils import row_col_sort, enlarge_bbox, check_inside, intersect_iou


class ActionEvaluator(object):
    BBOX_PATTERN = re.compile(r'\[ *(\d+) *, *(\d+) *, *(\d+) *, *(\d+) *\]')

    def __init__(self, demo_mode, screen_mode) -> None:
        self.demo_mode = demo_mode
        self.screen_mode = screen_mode

        self._all_action_types_ = [
            "click", 
            "scroll", 
            "type", 
            "press",
            "stop"
        ]
    
    def action_map(self, action_api:str):
        if not action_api: return None
        action_api = action_api.lower()
        if action_api == "input": return "type"
        for act_type in self._all_action_types_:
            if act_type in action_api: return act_type
        return None

    def _parse_action_(self, pred, w, h):
        pr = pred.get('action_predict', {})
        if self.demo_mode not in pr: return (None,) * 6

        action = pr[self.demo_mode].get(self.screen_mode, {})
        if not action: return (None,) * 6

        pd_action_type = self.action_map(action.get('ACTION', None))
        # if pd_action_type is None: print(action)

        pd_action_args = action.get('ARGS', {})
        if not isinstance(pd_action_args, dict): pd_action_args = {}
        pd_action_bbox = pd_action_args.get('bbox', None)
        if pd_action_bbox is not None:
            xmin, ymin, xmax, ymax = pd_action_bbox[:4]
            xmin = round(xmin/1000 * w)
            ymin = round(ymin/1000 * h)
            xmax = round(xmax/1000 * w)
            ymax = round(ymax/1000 * h)
            pd_action_bbox = [xmin, ymin, xmax, ymax]

        pd_action_idx = pd_action_args.get('idx', None)
        if pd_action_idx: 
            try: pd_action_idx = int(pd_action_idx)
            except: pd_action_idx = None
        pd_action_direction = pd_action_args.get('direction', None)
        pd_action_text = pd_action_args.get('text', "")
        pd_action_button = None if pd_action_type != "press" else \
            action['ACTION'].split("_")[1].lower()
        
        return pd_action_type, pd_action_bbox, pd_action_idx, \
               pd_action_direction, pd_action_text, pd_action_button

    def _parse_answer_(self, gt):
        gt_words = gt['coat_action_desc'].split(' ')

        gt_action_type = self.action_map(gt_words[0])
        if gt_action_type is None: print(gt['subset'], gt['episode_id'])
        gt_action_text = gt['result_action_text']
        gt_action_direction = "" if gt_action_type != "scroll" else gt_words[1].strip()
        gt_action_button = ""
        if gt_action_type == "press":
            for button in ['enter', 'back', 'home']: 
                if button in gt['coat_action_desc']: 
                    gt_action_button = button
                    break

        w, h = imagesize.get(gt['image_full_path'])
        gt_action_xy = [0, 0]
        if gt_action_type == "scroll":
            rel_y, rel_x = json.loads(gt['result_touch_yx'])
            abs_y, abs_x = int(rel_y*h), int(rel_x*w)
            gt_action_xy = [abs_x, abs_y]
        
        gt_cand_nodes = []
        for org_bbox, txt, ui_class in zip(gt['ui_positions'], gt['ui_text'], gt['ui_types']):
            ymin, xmin, h, w  = org_bbox
            bbox = [xmin, ymin, xmin+w, ymin+h]
            gt_cand_nodes.append({"bounds": bbox, "text": txt, "type": ui_class})
        gt_cand_nodes = row_col_sort(gt_cand_nodes)
        
        return gt_action_type, gt_action_xy, gt_cand_nodes, \
               gt_action_text, gt_action_button, gt_action_direction

    def _check_click_(self, pred_bbox, gt_xy, gt_nodes):
        # gt_xy is within pred_bbox
        if not pred_bbox: return False
        pred_bbox = enlarge_bbox([pred_bbox])[0]
        xmin, ymin, xmax, ymax = pred_bbox
        gt_x, gt_y = gt_xy
        is_correct = (xmin <= gt_x <= xmax and ymin <= gt_y <= ymax)
        if is_correct: return True

        # gt_xy is within any bbox
        bbox_array = enlarge_bbox([x['bounds'] for x in gt_nodes], scale_factor=1.2)
        is_inside, bbox_inside = check_inside(gt_x, gt_y, bbox_array)
        if is_inside:
            ious = intersect_iou(pred_bbox, bbox_inside)
            if np.any(ious > 0.5): return True

        return False

    def __call__(self, gt, pred):
        """ eval_single_step """
        subset, episode_id, step_id = gt['subset'], gt['episode_id'], gt['step_id']
        w, h = imagesize.get(gt['image_full_path'])
        
        # get ground truth information
        gt_action_type, gt_action_xy, gt_cand_nodes, \
            gt_action_text, gt_action_button, gt_action_direction = self._parse_answer_(gt)
        if not gt_action_type: print(gt['coat_action_desc'])
        gt_action_detail = {
            "click": gt_action_xy, "scroll": gt_action_direction, 
            "type": gt_action_text, "press": gt_action_button, "stop": "stop"
        }.get(gt_action_type, None)
        
        # get predict action information
        pd_action_type, pd_action_bbox, pd_action_idx, \
            pd_action_direction, pd_action_text, pd_action_button = self._parse_action_(pred, w, h)

        # compute metrics
        hit_format = True if pd_action_type is not None else False
        type_match = (pd_action_type is not None and gt_action_type == pd_action_type)

        exact_match = False
        pd_action_detail = None
        text_dist = None
        if type_match and pd_action_type == "click":
            if self.screen_mode == "tag" and pd_action_idx: # transform idx into bbox
                if 0 <= pd_action_idx < len(gt_cand_nodes):
                    pd_action_bbox = gt_cand_nodes[pd_action_idx]['bounds']
            pd_action_detail = pd_action_bbox
            exact_match = self._check_click_(pd_action_bbox, gt_action_xy, gt_cand_nodes)
        
        if type_match and pd_action_type == "scroll":
            pd_action_detail = pd_action_direction
            exact_match = (pd_action_direction == gt_action_direction)

        if type_match and pd_action_type == "type":
            pd_action_detail = pd_action_text
            text_dist = Levenshtein.ratio(pd_action_text, gt_action_text)
            exact_match = (pd_action_text in gt_action_text or \
                           gt_action_text in pd_action_text or \
                           text_dist > 0.8)

        if type_match and pd_action_type == "press":
            pd_action_detail = pd_action_button
            exact_match = (pd_action_button == gt_action_button)

        if type_match and pd_action_type == "stop":
            pd_action_detail = "stop"
            exact_match = True

        return {
            "subset": subset, "episode_id": episode_id, "step_id": step_id,
            "answer": {"action_type": gt_action_type, "action_detail": gt_action_detail},
            "pred": {"action_type": pd_action_type, "action_detail": pd_action_detail},
            "type_match": type_match, "exact_match": exact_match, 
            "text_dist": text_dist, "format_hit": hit_format
        }

    def compute_episode_metrics(self, episode_results):
        success, progress = [], []
        for __, eplist in episode_results.items():
            ep_success, ep_progress = True, 0
            for ex in eplist:
                if ex['exact_match'] is True: ep_progress += 1
                else: ep_success = False
                if not ep_success: break
            success.append(ep_success)
            progress.append(ep_progress/len(eplist)*1.0)
        
        return {"success_rate": round(sum(success) / len(success), 4),
                "goal_progress": round(sum(progress) / len(progress), 4)}

    def compute_atomic_metrics(self, step_results):
        recorder = {
            'total':  {'count':0, 'type_match':0, 'exact_match':0, "hit": 0}, 
            # -------------------------------------------
            'CLICK':  {'count':0, 'type_match':0, 'exact_match':0},  
            'TYPE':   {'count':0, 'type_match':0, 'exact_match':0, 'text_dist': []},
            'SCROLL': {'count':0, 'type_match':0, 'exact_match':0},  
            'PRESS':  {'count':0, 'type_match':0, 'exact_match':0},  
            'STOP':   {'count':0, 'type_match':0, 'exact_match':0}, 
        }

        for step in step_results:
            recorder['total']['count'] += 1
            recorder['total']['hit'] += step['format_hit']
            
            action_type = step['answer']['action_type'].upper()
            recorder[action_type]['count'] += 1
            recorder[action_type]['type_match'] += step['type_match']
            recorder['total']['type_match'] += step['type_match']
            recorder[action_type]['exact_match'] += step['exact_match']
            recorder['total']['exact_match'] += step['exact_match']
            if 'text_dist' in recorder[action_type] and step['text_dist'] is not None:
                recorder[action_type]['text_dist'].append(step['text_dist'])

        scores = {metric_key:{} for metric_key in ['total', 'CLICK', 'SCROLL', 'PRESS', 'STOP', 'TYPE']}
        scores['total']['hit_rate'] = round(recorder['total']['hit']/recorder['total']['count'], 4)
        for metric_key in ['total', 'CLICK', 'SCROLL', 'PRESS', 'STOP', "TYPE"]:
            scores[metric_key]['type_acc'] = round(recorder[metric_key]['type_match']/recorder[metric_key]['count'], 4)
            scores[metric_key]['exact_acc'] = round(recorder[metric_key]['exact_match']/recorder[metric_key]['count'], 4)
        if recorder['TYPE']['text_dist']:
            scores['TYPE']['text_dist'] = round(sum(recorder['TYPE']['text_dist'])/len(recorder['TYPE']['text_dist']), 4)
        return scores

