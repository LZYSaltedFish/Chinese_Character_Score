# Chinese_Character_Score
## 本科毕业设计 - 基于神经网络的笔迹提取与书写评分模型研究
### 说明

- 安装依赖:
```bash
pickle
matplotlib
opencv-python
numpy
tensorflow
```

- 视频输入目录:
```
test_video/
```

- 笔画分割结果输出目录
```
test_video/strokes
```

- 分割对比视频输出目录
```
test_video/result
```

- 运行测试
修改`core/core.py`中的`video_path`、`STROKE_NUM`及`get_char_id()`中的参数，然后执行
```bash
python core/core.py
```

- 修改参数
通过修改`core/core.py`中传入`get_strokes()`的参数，设置各选项

- 其他说明
    - `utils/character_feature_dict`和`utils/stroke_feature_dict`为预计算的汉字整字重心特征、网格特征字典文件