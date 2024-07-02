<div align="center">
  <h1 style="display: inline-block; font-size: 32px;">Android in the Zoo:<br>Chain-of-Action-Thought for GUI Agents</\br></h1>
</div>
<p align="center"><strong>Jiwen Zhang<sup>1,2</sup> , Jihao Wu<sup>2</sup>  , Yihua Teng<sup>2</sup>  , Minghui Liao<sup>2</sup>  , Nuo Xu<sup>2</sup>  , Xiao Xiao<sup>2</sup>  , Zhongyu Wei<sup>1</sup> , Duyu  Tang<sup>2</sup>.
 </strong></p>
<p align="center"><sup>1</sup>Fudan University      <sup>2</sup>Huawei Inc.</p> 
<p align="center">
    <img src="https://img.shields.io/badge/Version-v1.0-Green" />
    <img src="https://img.shields.io/badge/Licence-Apache_2.0-Green" />
    <img src="https://img.shields.io/github/stars/IMNearth/CoAT?label=Stars" />
    <a href="https://arxiv.org/abs/2403.02713"><img src="https://img.shields.io/badge/Paper-Arxiv-red" /></a>
    <a href="https://pan.baidu.com/s/1dHG-4L0RE1aYINzMSA4dCw?pwd=7g82">
      <img src="https://img.shields.io/badge/CoAT-Dataset-violet?logo=baidu" />
  	</a>
    <a href="https://drive.google.com/file/d/12xOV2m62fBUFLhMcWIsFiC6zwV7a2RhI/view?usp=drive_link">
      <img src="https://img.shields.io/badge/CoAT-Dataset-blue?logo=googledrive" />
  	</a>
</p> 

--------------

This work presents **Chain-of-Action-Thought** (dubbed **CoAT**), which takes the description of the previous actions, the current screen, and more importantly the action thinking of what actions should be performed and the outcomes led by the chosen action. To enable an adaptive learning of CoAT process, we construct a benchmark **Android-In-The-Zoo**, which contains 18,643 screen-action pairs together with CoAT annotations.

<div align="center">
    <img src=assets/intro-total.png width=80% />
</div>


## 📣 Update

- **[2024-03-31]** We release the first version of our AiTZ dataset!

- **[2024-03-05]** We have our paper arxived, now you can acess it by clicking [here](https://arxiv.org/abs/2403.02713) !



## Android-in-the-Zoo

The data in AiTZ has 18,643 screens together with 2500+ instructions, all annotated with CoAT-driven semantic labels. The sample format for each time step is

```json
{
  "episode_id": "523638528775825151",
  "episode_length": 4,
  "step_id": 0,
  "coat_screen_desc":   "[observe]",
  "coat_action_think":  "[action think]",
  "coat_action_desc":   "[next action description]",
  "coat_action_result": "[action result]",
  ...
}
```

You can refer to  `data-example` folder for a more specific example.



### Download

Our dataset ([GoogleDrive](https://drive.google.com/file/d/12xOV2m62fBUFLhMcWIsFiC6zwV7a2RhI/view?usp=drive_link) or [BaiduNetdisk](https://pan.baidu.com/s/1dHG-4L0RE1aYINzMSA4dCw?pwd=7g82)) contains both the screens (.png) and the annotations (.json), consuming about 2.6G device space. 

 

### Statistics

| Subset      | Train      |           | Test       |           |
| ----------- | ---------- | --------- | ---------- | --------- |
|             | \#Episodes | \#Screens | \#Episodes | \#Screens |
| General     | 323        | 2405      | 156        | 1202      |
| Install     | 286        | 2519      | 134        | 1108      |
| GoogleApps  | 166        | 1268      | 76         | 621       |
| Single      | 844        | 2594      | 0          | 0         |
| WebShopping | 379        | 5133      | 140        | 1793      |
| **Total**   | **1998**   | **13919** | **506**    | **4724**  |





## Citation

If you find our work helpful, please consider citing our paper.

```
@misc{zhang2024android,
      title={Android in the Zoo: Chain-of-Action-Thought for GUI Agents}, 
      author={Jiwen Zhang and Jihao Wu and Yihua Teng and Minghui Liao and Nuo Xu and Xiao Xiao and Zhongyu Wei and Duyu Tang},
      year={2024},
      eprint={2403.02713},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

