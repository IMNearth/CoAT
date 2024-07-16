import os
import json
import yaml
import time
import random
import pandas as pd
from collections import defaultdict
from yacs.config import CfgNode as CN
from tqdm import tqdm

from coat.model import get_model
from coat.action_utils import extract_gt_action
from coat.agent import ScreenAgent
import threading


class AITZDataset(object):
    """
    Creating a custom dataset for reading the processed AITW data
    with GPT-4V labeled detail semantic annotations.
    """
    DATASET_DIR = {
    'general': '{}/general',
    'google_apps': '{}/google_apps',
    'install': '{}/install',
    'single': '{}/single',
    'web_shopping': '{}/web_shopping',
    }

    def __init__(self, split="test", data_dir="android-in-the-zoo", ratio=1.0, double_sample=False) -> None:
        self.ratio = ratio
        self.double_sample = double_sample
        self.data_dir = os.path.join(data_dir, split)

        self.episode_data = self._load_data_()
        self.data = self._split_to_steps_(self.episode_data)

    def _load_data_(self): 
        valid_paths = defaultdict(list)
        for subset in self.DATASET_DIR:
            subdata_dir = self.DATASET_DIR[subset].format(self.data_dir)
            if os.path.exists(subdata_dir):
                sequence_names = os.listdir(subdata_dir)
                for seq_name in sequence_names:
                    seq_dir = os.path.join(subdata_dir, seq_name)
                    if not os.path.isdir(seq_dir): continue
                    episode_path = os.path.join(seq_dir, f"{seq_name}.json")
                    valid_paths[subset].append(episode_path)
        
        sampled_paths = []
        for subset, v_paths in valid_paths.items():
            N = len(v_paths)
            k=int(self.ratio*N)
            # random.sample(v_paths, k) if self.ratio < 1.0 else v_paths
            sampled_paths += random.sample(v_paths, k) if self.ratio < 1.0 else v_paths

        ep_data = []
        for episode_path in sampled_paths:
            episode_data = json.load(open(episode_path, "r"))                        
            ep_data.append(episode_data)
        return ep_data

    def _split_to_steps_(self, episode_data):
        data = []
        for edx, episode in enumerate(episode_data):
            history_plain_actions, history_coat_actions = [], []
            for idx, step in enumerate(episode):
                step['subset'] = step['image_path'].split('/')[0]
                step['image_full_path'] = os.path.join(self.data_dir, step['image_path'])
                step['prev_step_id'] = episode[idx-1]['step_id'] if idx > 0 else None
                next_img_path = os.path.join(self.data_dir, episode[idx+1]['image_path']) \
                    if idx + 1 < len(episode) else None
                step['next_image_full_path'] = next_img_path
                step['history_actions'] = history_plain_actions[:]
                step['history_coat_actions'] = history_coat_actions[:]
                step['result_action'] = extract_gt_action(step)[1]
                for ui_key in ['ui_positions', 'ui_text', 'ui_types']:
                    step[ui_key] = json.loads(step[ui_key])
                data.append(step)
                history_plain_actions.append(step['result_action'])
                history_coat_actions.append(step['coat_action_desc'])

        return data

    def __len__(self, ): return len(self.data)

    def __getitem__(self, index): return self.data[index]

# ==========================================================================================

def sinle_run_flow(agent:ScreenAgent, step_data:dict, save_dir:str):
    agent.flow(step_data, save_dir=save_dir)


def sinle_run_predict(agent:ScreenAgent, step_data:dict, save_dir:str):
    agent.predict(step_data, save_dir=save_dir)


def collect(cfg, num_threads=2, task="flow"):
    # cfg.MODEL.NAME = "openai"
    if task == "flow": todo_task = sinle_run_flow 
    if task == "predict": todo_task = sinle_run_predict
    
    if cfg.MODEL.NAME in ["gemini", "openai"]:
        print("using proxy ... ")
        os.environ['http_proxy'] = "your_proxy"
        os.environ['https_proxy'] = "your_proxy"
    
    aitz = AITZDataset(cfg.DATA.SPLIT, cfg.DATA.DATA_DIR, ratio=0.1, double_sample=False)
    print(len(aitz), len(aitz.episode_data))

    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME)
    agent = ScreenAgent(config=cfg)

    before_count = threading.active_count() + 1
    threads = []
    last_time = time.time()
    pbar = tqdm(aitz.data)
    for idx, step_data in enumerate(pbar):
        thread = threading.Thread(target=todo_task, args=(agent, step_data, save_dir))
        time.sleep(max(0.01-(time.time()-last_time), 0))
        thread.start()
        last_time = time.time()
        threads.append(thread)
        pbar.set_description(f"Active threads [{threading.active_count()-before_count}]")
        while threading.active_count() - before_count >= num_threads: time.sleep(1)

        if len(threads) == 100: 
            for thr in threads: thr.join()
            threads = []
    
    print()
    while threading.active_count() - before_count >0: time.sleep(1)
    for thr in threads: thr.join()

# ==========================================================================================

def try_model(cfg):
    cfg.MODEL.NAME = "qwenvl"

    if cfg.MODEL.NAME in ["gemini", "openai"]:
        os.environ['http_proxy'] = "your_proxy"
        os.environ['https_proxy'] = "your_proxy"
    
    prompt = "Describe the image."
    image_path = "android-in-the-wild/aitw_with_gpt/test/general/GENERAL-1359994677477286277/GENERAL-1359994677477286277_2.png"
    model = get_model(cfg.MODEL, seed=2024)
    res_json, res_state = model.get_response(image_path, prompt)
    print(res_json, res_state)
    pass

# ==========================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoAT Demo")
    parser.add_argument("--config-file", default="coat/config.yaml", metavar="FILE",  help="path to config file",)
    parser.add_argument("--task", default="eval", type=str, choices=['try', 'flow', 'predict', 'eval'])
    parser.add_argument("--num-threads", default=1, type=int, help="number of threads")
    parser.add_argument("--seed", default=2020, type=int, help="random seed")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER, )
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    cfg = CN(yaml.safe_load(open(args.config_file)))    
    cfg.merge_from_list(args.opts)

    if args.task == "try": 
        try_model(cfg)
    if args.task in ['flow', 'predict']: 
        collect(cfg, num_threads=args.num_threads, task=args.task)

