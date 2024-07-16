import os
import time
import json
import traceback
import imagesize
import numpy as np
from PIL import Image
from http import HTTPStatus
from .model import get_model
from .screen_utils import row_col_sort, draw_bbox


def json_parser(json_string:str):
    if json_string.startswith("```json"):
        json_string = json_string[7:]
    if json_string.startswith("```JSON"):
        json_string = json_string[7:]
    if json_string.endswith("```"):
        json_string = json_string[:-3]
    
    if "```json" in json_string: 
        json_string = json_string.split("```json")[1]
    if "```JSON" in json_string: 
        json_string = json_string.split("```JSON")[1]
    if "```" in json_string:
        json_string = json_string.split("```")[0]
    
    return json.loads(json_string)



class BaseAgent(object):
    
    def __init__(self, config, *args, **kwargs) -> None:
        self.cfg = config
        self.prompts = config['PROMPTS']
        self.vlm = get_model(config['MODEL'])

    def observe(self, user_request, screen_path, **kwargs):
        sys_rompt = self.prompts['SCREEN_DESC']['SYSTEM']
        usr_prompt = self.prompts['SCREEN_DESC']['USER']
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
        
        res, state = self.vlm.get_response(screen_path, prompt)
        return res, state

    def think_action(self, user_request, screen_path, screen_desc, history_actions, **kwargs):
        sys_rompt = self.prompts['ACTION_THINK_DESC']['SYSTEM']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        
        usr_prompt = self.prompts['ACTION_THINK_DESC']['USER']
        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x.strip() for x in usr_prompt.split("{screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0})
        ]
        for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))

        res, state = self.vlm.get_response(screen_path, prompt)
        return res, state

    def reflect_result(self, user_request, screen_path, next_screen_path, last_action, **kwargs):
        sys_rompt = self.prompts['ACTION_RESULT']['SYSTEM']
        
        usr_prompt = self.prompts['ACTION_RESULT']['USER']
        usr_prompt = usr_prompt.replace("{last_action}", last_action)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)

        usr_prompt = [x.strip() for x in usr_prompt.split("{before_screenshot}")]
        usr_prompt = [usr_prompt[0]] + [x.strip() for x in usr_prompt[1].split("{after_screenshot}")]

        prompt = [
            ("system", sys_rompt),
            ("user", {"text": usr_prompt[0], "img_index": 0}),
            ("user", {"text": usr_prompt[1], "img_index": 1})
        ]
        for txt in usr_prompt[2:]: prompt.append(("user", {"text": txt}))
        
        res, state = self.vlm.get_response([screen_path, next_screen_path], prompt)
        return res, state

    def flow(self, step_data, save_dir, max_trials=5):
        subset, episode_id, step_id = step_data['subset'], step_data['episode_id'], step_data['step_id']
        user_request, image_path = step_data['instruction'], step_data['image_full_path']

        save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        if not os.path.exists(save_path): json.dump({}, open(save_path, "w", encoding="utf-8"))

        prev_anno = json.load(open(save_path, "r"))
        # if 'action_result' not in prev_anno: return
        # del prev_anno['action_result']
        # json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        # return

        update = False
        if 'screen_desc' not in prev_anno: 
            print(f"Genearting [screen_desc] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                res, state = self.observe(user_request, image_path)
                if state == HTTPStatus.OK: 
                    prev_anno['screen_desc'], update = res.strip(), True
                    break
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [screen_desc] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
        update = False
        if 'action_think' not in prev_anno:
            print(f"Genearting [action_think] -> img_path {image_path}")
            try_num = 0
            while try_num < max_trials:
                res, state = self.think_action(user_request, image_path, 
                                               screen_desc=prev_anno['screen_desc'], 
                                               history_actions=step_data['history_coat_actions'])
                if state == HTTPStatus.OK: 
                    try: 
                        response = json_parser(res)
                        action_think = response['Thought']
                        action_plan = response['Future Action Plan']
                        action_desc = response['Next Single Step Action']
                        prev_anno['action_think'] = action_think
                        prev_anno['action_plan'] = action_plan
                        prev_anno['action_desc'] = action_desc
                        update = True
                        break
                    except Exception as e: 
                        if not isinstance(e, json.decoder.JSONDecodeError): print(traceback.format_exc())
                        else: print(res,"\n", save_path, "\n", sep="")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(5)
        if update: print(f"Updating [action_think] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)
        
        update = False
        if 'action_result' not in prev_anno:
            print(f"Genearting [action_result] -> img_path {image_path}")
            if step_data['result_action_type'] not in [10, 11]:
                try_num = 0
                cur_image_path, next_image_path = image_path, step_data['next_image_full_path']
                while try_num < max_trials:
                    cur_image_height = imagesize.get(cur_image_path)[1]
                    next_image_height = imagesize.get(next_image_path)[1]
                    res, state = self.reflect_result(user_request, 
                                                     screen_path=cur_image_path, 
                                                     next_screen_path=next_image_path, 
                                                     last_action=step_data['coat_action_desc'])
                    if state == HTTPStatus.OK: 
                        prev_anno['action_result'], update = res, True
                        break
                    elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
                        cur_image_path = self.compress_image(image_path, max_height=int(0.8*cur_image_height))
                        next_image_path = self.compress_image(step_data['next_image_full_path'], max_height=int(0.8*next_image_height))
                    else: print(res)
                    print(f"Trial {try_num} failured! -- Status {state}")
                    try_num += 1
                    time.sleep(5)
            else: 
                prev_anno['action_result'] = "The execution of user request is stopped."
                update = True
        if update: print(f"Updating [action_result] {save_path} ...")
        if update: json.dump(prev_anno, open(save_path, "w", encoding="utf-8"), indent=4)

    def compress_image(self, image_path, max_height=1024, mode="scale"):
        org_img_path, ext = os.path.splitext(image_path)

        screen_img = Image.open(image_path)
        image_width, image_height = screen_img.size

        if image_height > max_height:
            scale_percent = max_height / image_height
            height = int(scale_percent * image_height)
            width = int(scale_percent * image_width)
            screen_img = screen_img.resize((width, height))
        
        if mode == "scale":
            save_img_path = org_img_path+"_tmp"+ext
            screen_img.save(save_img_path)
        elif mode == "jpg":
            screen_img = screen_img.convert("RGB")
            save_img_path = org_img_path+"_tmp"+".jpeg"
            screen_img.save(save_img_path)
        else:
            save_img_path = image_path

        return save_img_path


class ScreenAgent(BaseAgent):

    def __init__(self, config) -> None:
        super().__init__(config)

        self.use_screen_tag = self.cfg.DATA.USE_SCREEN_TAG
        self.use_screen_txt = (not self.use_screen_tag)
        self.screen_mode = "txt" if self.use_screen_txt else "tag"
        pass
    
    def format_bbox(self, bbox_xyxy, width, height):
        coord_sys, rep_type = self.cfg.DATA.BBOX_FORMAT.split("_")
        
        if coord_sys == "relative":
            bbox = np.array(bbox_xyxy) / np.array([width, height, width, height])
            if rep_type == "int":
                bbox = (bbox*1000).astype(np.int32).tolist()
            else:
                bbox = [round(x, 3) for x in bbox.tolist()]
        elif coord_sys == "absolute": bbox = bbox_xyxy
        else: raise NotImplementedError
        box_str = json.dumps(bbox)
        return box_str

    def add_screen_tag(self, step_data, save_dir, update=False):
        """ Set of Mark Prompting """
        src_image_path = step_data['image_full_path']
        base_name, ext = os.path.splitext(src_image_path)
        dst_image_path = base_name + "_tag" + ext
        # image_name = os.path.basename(src_image_path)
        # dst_image_path = os.path.join(save_dir, image_name)

        if not os.path.exists(dst_image_path) or update:
            ui_bboxes = []
            for ui_node in step_data['ui_elements']:
                ui_bboxes.append(ui_node['bounds'])
            
            tag_image = draw_bbox(src_image_path, bboxes=ui_bboxes, 
                                texts=list(map(str, range(len(step_data['ui_positions'])))),
                                rgba_color=(255, 0, 0, 180), thickness=5)
            tag_image.save(dst_image_path)
        return dst_image_path

    def add_screen_txt(self, step_data, indent=2): 
        """ Textual Representation of Screen Elements """
        image_path = step_data['image_full_path']
        w, h = imagesize.get(image_path)

        screen_str = []
        for ui_node in step_data['ui_elements']:
            ui_str = " "*indent + ui_node['type'].upper()
            if ui_node['text']: ui_str += " " + ui_node['text']
            bbox_str = self.format_bbox(ui_node['bounds'], w, h)
            ui_str += " " + bbox_str
            screen_str.append(ui_str)
        
        return "\n".join(screen_str)

    def make_action(self, step_data, usr_prompt, save_dir, asst_prompt=None):
        action_mode = self.cfg.DEMO_MODE.upper()
        image_path = step_data['image_full_path']

        ui_elements = []
        for org_bbox, txt, ui_class in zip(
            step_data['ui_positions'], step_data['ui_text'], step_data['ui_types']):
            ymin, xmin, h, w  = org_bbox
            bbox = [xmin, ymin, xmin+w, ymin+h]
            ui_elements.append({"bounds": bbox, "text": txt, "type": ui_class})
        ui_elements = row_col_sort(ui_elements)
        step_data['ui_elements'] = ui_elements

        if self.use_screen_tag:
            sys_prompt = self.prompts['ACTION_PREDICT'][action_mode]['SYSTEM']['SCREEN_TAG']
            action_space = self.prompts['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TAG']
            sys_prompt = sys_prompt.replace("{action_space}", action_space)
            tag_image_path = self.add_screen_tag(step_data, save_dir=save_dir)
            
            if self.cfg.MODEL.NAME == "openai":
                image_path = self.compress_image(image_path, max_height=1440, mode="jpg")
                tag_image_path = self.compress_image(tag_image_path, max_height=1440, mode="scale")

            prompt = [
                ("system", sys_prompt),
                ("user", {"text": usr_prompt[0], "img_index": 0}),
                ("user", {"text": "", "img_index": 1})
            ]
            for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
            if asst_prompt: prompt.append(("assistant", {"text": asst_prompt}))
            asst_res, state = self.vlm.get_response([image_path, tag_image_path], prompt)
        
        if self.use_screen_txt:
            sys_prompt = self.prompts['ACTION_PREDICT'][action_mode]['SYSTEM']['SCREEN_TXT']
            action_space = self.prompts['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TXT']
            sys_prompt = sys_prompt.replace("{action_space}", action_space)
            screen_txt = self.add_screen_txt(step_data)
            
            prompt = [
                ("system", sys_prompt),
                ("user", {"text": usr_prompt[0], "img_index": 0}),
                ("user", {"text": f"<SCREEN ELEMENTS>: {screen_txt}"})
            ]
            for txt in usr_prompt[1:]: prompt.append(("user", {"text": txt}))
            if asst_prompt: prompt.append(("assistant", {"text": asst_prompt}))
            asst_res, state = self.vlm.get_response([image_path], prompt)

        asst_head = asst_prompt if asst_prompt else ""         
        return asst_head, asst_res, state

    def coa_action(self, step_data, *args, **kwargs):
        """ Chain of Action """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_actions']

        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."

        usr_prompt = self.prompts['ACTION_PREDICT']['COA']['USER']
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'])
 
    def cot_action(self, step_data, *args, **kwargs):
        """ Chain of Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        acton_think = kwargs['action_think']
        
        usr_prompt = self.prompts['ACTION_PREDICT']['COT']['USER']
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        if self.cfg.MODEL.NAME == "qwenvl":
            asst_prompt = json.dumps({"THINK": acton_think}, indent=4)
        else:
            asst_prompt = self.prompts['ACTION_PREDICT']['COT']['ASST']
            asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

    def coat_action(self, step_data, *args, **kwargs):
        """ Chain of Action Thought """
        user_request, image_path = step_data['instruction'], step_data['image_full_path']
        history_actions = step_data['history_coat_actions']
        
        screen_desc = kwargs['screen_desc']
        prev_action_result = kwargs.get('prev_action_result', None)
        acton_think, next_action_desc = kwargs['action_think'], kwargs['action_desc']

        usr_prompt = self.prompts['ACTION_PREDICT']['COAT']['USER']
        usr_prompt = usr_prompt.split("\n")
        if not history_actions: usr_prompt = [x for x in usr_prompt if 'history_actions' not in x]
        if not prev_action_result: usr_prompt = [x for x in usr_prompt if 'prev_action_result' not in x]
        usr_prompt = "\n".join(usr_prompt)

        usr_prompt = usr_prompt.replace("{screen_desc}", screen_desc)
        if history_actions: history_actions = ", ".join(history_actions)
        else: history_actions = "None"
        history_actions = history_actions + "."
        usr_prompt = usr_prompt.replace("{history_actions}", history_actions)
        if prev_action_result:
            usr_prompt = usr_prompt.replace("{prev_action_result}", prev_action_result)
        usr_prompt = usr_prompt.replace("{user_request}", user_request)
        usr_prompt = [x for x in usr_prompt.split("{screenshot}")]

        if self.cfg.MODEL.NAME == "qwenvl":
            asst_prompt = json.dumps({"THINK": acton_think, "NEXT": next_action_desc,}, indent=4)
        else:
            asst_prompt = self.prompts['ACTION_PREDICT']['COAT']['ASST']
            asst_prompt = asst_prompt.replace("{action_thought}", json.dumps(acton_think))
            asst_prompt = asst_prompt.replace("{next_single_action}", json.dumps(next_action_desc))

        return self.make_action(step_data, usr_prompt, kwargs['save_dir'], asst_prompt=asst_prompt)

    def predict(self, step_data, save_dir, max_trials=5): 
        subset, episode_id = step_data['subset'], step_data['episode_id']
        prev_step_id, step_id = step_data['prev_step_id'], step_data['step_id']

        save_dir = os.path.join(save_dir, f"{subset}-{episode_id}")
        
        cur_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{step_id}.json")
        cur_anno = json.load(open(cur_save_path, "r"))
        if prev_step_id:
            prev_save_path = os.path.join(save_dir, f"{subset}-{episode_id}_{prev_step_id}.json")
            prev_anno = json.load(open(prev_save_path, "r"))
            cur_anno['prev_action_result'] = prev_anno['action_result']

        action_mode = self.cfg.DEMO_MODE.upper()
        if action_mode == "COA":    func_handler = self.coa_action
        elif action_mode == "COT":  func_handler = self.cot_action
        elif action_mode == "COAT": func_handler = self.coat_action
        else: raise NotImplementedError

        if 'action_predict' not in cur_anno: cur_anno['action_predict'] = {}
        if action_mode not in cur_anno['action_predict']: cur_anno['action_predict'][action_mode] = {}
        # if action_mode in cur_anno['action_predict']: 
        #     anno = cur_anno['action_predict'][action_mode]
        #     cur_anno['action_predict'][action_mode] = {}
        #     cur_anno['action_predict'][action_mode][self.screen_mode] = anno
        #     json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
        
        if self.screen_mode not in cur_anno['action_predict'][action_mode]:
            print(f"[Mode {action_mode}][{self.screen_mode}] Gnerating action ... ({cur_save_path})")
            try_num = 0
            while try_num < max_trials:
                asst_head, asst_res, state = func_handler(step_data, **cur_anno, save_dir=save_dir)
                if state == HTTPStatus.OK: 
                    try: 
                        try: response = json_parser(asst_res)
                        except json.decoder.JSONDecodeError as e:
                            try: response = json_parser(asst_head + asst_res)
                            except: 
                                try:response = json_parser(asst_res + "}")
                                except: response = json_parser(asst_head + asst_res + "}")
                        cur_anno['action_predict'][action_mode][self.screen_mode] = {
                            "ACTION": response["ACTION"], "ARGS": response["ARGS"]}
                        
                        print(f"Updating [{action_mode}][{self.screen_mode}] {cur_save_path} ...")
                        json.dump(cur_anno, open(cur_save_path, "w", encoding="utf-8"), indent=4)
                    except json.decoder.JSONDecodeError as e:
                        print('-'*50 + "\n", asst_head, sep=" ")
                        print('-'*50 + "\n", asst_res, sep=" ")
                    break
                elif state == HTTPStatus.REQUEST_ENTITY_TOO_LARGE: 
                    print(f"Trial {try_num} failured! -- Status {state}")
                    break
                print(f"Trial {try_num} failured! -- Status {state}")
                try_num += 1
                time.sleep(10)

        return None
