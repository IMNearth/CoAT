from yacs.config import CfgNode as CN

_C = CN()
_C.DEMO_NAME = ""       # 当前实验的name
_C.DEMO_MODE = ""       # demo模式
_C.OUTPUT_DIR = ""      # demo输出的保存路径


_C.DATA = CN()          # ----- 数据集相关配置  ----- 
_C.DATA.DATA_DIR = ""   # 数据集路径
_C.DATA.SPLIT = ""      # 进行测试的数据集划分


_C.MODEL = CN()         # ----- 模型API相关配置 ----- 
_C.MODEL.NAME = ""
_C.MODEL.OPENAI_API_KEY = ""
_C.MODEL.OPENAI_API_URL = ""
_C.MODEL.GEMINI_API_KEY = ""
_C.MODEL.GEMINI_MODEL = "gemini-pro-vision"
_C.MODEL.DASHSCOPE_API_KEY = ""
_C.MODEL.DASHSCOPE_MODEL = "qwen-vl-max"


_C.PROMPTS = CN()       # ----- 提示词相关配置 ----- 

_C.PROMPTS.SCREEN_DESC = CN()
_C.PROMPTS.SCREEN_DESC.SYSTEM = ""
_C.PROMPTS.SCREEN_DESC.USER = ""

_C.PROMPTS.ACTION_THINK = CN()
_C.PROMPTS.ACTION_THINK.SYSTEM = ""
_C.PROMPTS.ACTION_THINK.USER = ""

_C.PROMPTS.ACTION_DESC = CN()
_C.PROMPTS.ACTION_DESC.SYSTEM = ""
_C.PROMPTS.ACTION_DESC.USER = ""

_C.PROMPTS.ACTION_PREDICT = CN()
_C.PROMPTS.ACTION_PREDICT.SYSTEM = ""
_C.PROMPTS.ACTION_PREDICT.USER = ""

_C.PROMPTS.ACTION_RESULT = CN()
_C.PROMPTS.ACTION_RESULT.SYSTEM = ""
_C.PROMPTS.ACTION_RESULT.USER = ""


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
