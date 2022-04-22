# [Token Shift Transformer for Video Classification](https://paperswithcode.com/paper/token-shift-transformer-for-video)
## Reimplementation based on [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) 


## 1. 简介
![Token Shift Transformer](https://github.com/zwtu/TokShift-Transformer-Paddle/blob/main/images/model_structure.png?raw=true)
<strong>Paper：</strong> Zhang H, Hao Y, Ngo C W. [Token shift transformer for video classification]((https://paperswithcode.com/paper/token-shift-transformer-for-video))[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 917-925.
<strong>Code Reference：</strong>[https://github.com/VideoNetworks/TokShift-Transformer](https://github.com/VideoNetworks/TokShift-Transformer)   
<strong>复现目标：</strong>UCF101数据集，ImageNet-21k预训练模型条件下，8x256x256输入尺寸，Top1=91.65  


## 2. 复现精度
| Model <br> <br>    |Pretrain  <br> <br>    |  Res <br>(𝐻 × 𝑊 ) | # Frames <br> 𝑇  | UCF101 <br> Acc1 (%)   
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------:
| TokShift | ImageNet-21k | 256 × 256 | 8 | <strong>92.81</strong> |

## 3. 数据集和预训练模型

1. 下载数据集 [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) or [[PaddleVideo] UCF101视频分类数据集](https://aistudio.baidu.com/aistudio/datasetdetail/105621)，存放路径 ``` data/ucf101/ ```

2. 下载 raw annotations 并生成所需格式的 video annotations

    ```
    cd ~/data/ucf101/
    bash download_annotations.sh
    python build_ucf101_file_list.py ~/PaddleVideo-develop/data/ucf101/UCF-101/ --level 2 --format videos --out_list_path ./
    ```

3. <strong>optional</strong> 视频提帧 生成所需格式的 frame annotations
    ```
    cd ~/data/ucf101/
    !python extract_rawframes.py ./videos/ ./rawframes/ --level 2 --ext avi
    !python build_ucf101_file_list.py rawframes/ --level 2 --format rawframes --out_list_path ./
    ```

4. 下载预训练权重

    ```
    wget -P data/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
    ```


## 4. 环境依赖

- GPU：Tesla V100 32G
- Framework：PaddlePaddle == 2.2.2
-  ``` pip install -r requirements.txt ```


## 5. 快速开始

1. Clone 本项目

    ```
    git clone https://github.com/zwtu/TokShift-Transformer-Paddle.git
    cd TokShift-Transformer-Paddle
    ```

2. 模型训练

- 参数配置文件在 ``` configs/recognition/ ```

    ```
    python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --validate --seed=1234
    ```

- 部分训练日志：
    ```
    /home/aistudio/PaddleVideo-develop
    Loading weights
    [04/22 06:53:17] Training in amp mode, amp_level=O1.
    [04/22 09:14:09] epoch:[  9/25 ] train step:0    loss: 0.11621 lr: 0.060900 top1: 1.00000 top5: 1.00000 batch_cost: 5.48150 sec, reader_cost: 4.09958 sec, ips: 2.91891 instance/sec.
    [04/22 09:14:35] epoch:[  9/25 ] train step:20   loss: 0.11826 lr: 0.060900 top1: 1.00000 top5: 1.00000 batch_cost: 1.25546 sec, reader_cost: 0.00032 sec, ips: 12.74437 instance/sec.
    [04/22 09:26:03] epoch:[  9/25 ] train step:560  loss: 0.11194 lr: 0.060900 top1: 1.00000 top5: 1.00000 batch_cost: 1.25413 sec, reader_cost: 0.00023 sec, ips: 12.75788 instance/sec.
    [04/22 09:26:28] epoch:[  9/25 ] train step:580  loss: 0.11695 lr: 0.060900 top1: 1.00000 top5: 1.00000 batch_cost: 1.27711 sec, reader_cost: 0.00705 sec, ips: 12.52824 instance/sec.
    [04/22 09:26:48] END epoch:9   train loss_avg: 0.13345  top1_avg: 0.99643 top5_avg: 1.00000 avg_batch_cost: 1.25301 sec, avg_reader_cost: 0.00024 sec, batch_cost_sum: 764.19230 sec, avg_ips: 12.47853 instance/sec.
    [04/22 09:26:50] epoch:[  9/25 ] val step:0    loss: 0.01246 top1: 1.00000 top5: 1.00000 batch_cost: 2.08060 sec, reader_cost: 0.00000 sec, ips: 1.92253 instance/sec.
    [04/22 09:26:56] epoch:[  9/25 ] val step:20   loss: 0.02041 top1: 1.00000 top5: 1.00000 batch_cost: 0.30059 sec, reader_cost: 0.00000 sec, ips: 13.30695 instance/sec.
    [04/22 09:31:42] epoch:[  9/25 ] val step:920  loss: 1.04484 top1: 0.25000 top5: 1.00000 batch_cost: 0.29999 sec, reader_cost: 0.00000 sec, ips: 13.33392 instance/sec.
    [04/22 09:31:48] epoch:[  9/25 ] val step:940  loss: 0.19421 top1: 1.00000 top5: 1.00000 batch_cost: 0.29935 sec, reader_cost: 0.00000 sec, ips: 13.36221 instance/sec.
    [04/22 09:31:49] END epoch:9   val loss_avg: 0.30084 top1_avg: 0.92019 top5_avg: 0.98547 avg_batch_cost: 0.22932 sec, avg_reader_cost: 0.00000 sec, batch_cost_sum: 301.50225 sec, avg_ips: 12.55049 instance/sec.
    [04/22 09:31:51] Already save the best model (top1 acc)0.9201
    [04/22 14:20:25] END epoch:25  val loss_avg: 0.31817 top1_avg: 0.91702 top5_avg: 0.98547 avg_batch_cost: 0.23006 sec, avg_reader_cost: 0.00000 sec, batch_cost_sum: 316.87528 sec, avg_ips: 11.94161 instance/sec.
    [04/22 14:20:26] training TokenShift_ucf101_256_16_256_aug0.1_0.0609 finished
    ```

    训练完成后，模型参数保存至 ``` output/```

3. 模型评估

    ```
    !python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --test --seed=1234 -w 'output/TokenShift_ucf101_256_16_256_aug0.1_0.0609/TokenShift_ucf101_256_16_256_aug0.1_0.0609_best.pdparams'
    ```

- 部分测试日志：
    ```
    W0422 14:31:53.689891 31948 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0422 14:31:53.695667 31948 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [04/22 14:32:06] [TEST] Processing batch 0/945 ...
    [04/22 14:32:06] [TEST] Processing batch 1/945 ...
    [04/22 14:32:07] [TEST] Processing batch 2/945 ...
    [04/22 14:32:07] [TEST] Processing batch 3/945 ...
    [04/22 14:32:08] [TEST] Processing batch 4/945 ...
    [04/22 14:32:08] [TEST] Processing batch 5/945 ...
    [04/22 14:40:19] [TEST] Processing batch 940/945 ...
    [04/22 14:40:19] [TEST] Processing batch 941/945 ...
    [04/22 14:40:20] [TEST] Processing batch 942/945 ...
    [04/22 14:40:20] [TEST] Processing batch 943/945 ...
    [04/22 14:40:21] [TEST] Processing batch 944/945 ...
    [04/22 14:40:21] [TEST] Processing batch 945/945 ...
    [04/22 14:40:21] [TEST] finished, avg_acc1= 0.9281184077262878, avg_acc5= 0.9912790656089783 
    ```

4. 模型预测

- 模型动转静推理
    ```
    python3 tools/export_model.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -p 'output/TokenShift_ucf101_256_16_256_aug0.1_0.0609/TokenShift_ucf101_256_16_256_aug0.1_0.0609_best.pdparams'   
    ```
    在默认路径```inference/```下，生成三个对应文件
    <br>

- 模型静态推理
    ```
    python3 tools/predict.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -i 'BrushingTeeth.avi' --model_file ./inference/TokenShiftVisionTransformer.pdmodel --params_file ./inference/TokenShiftVisionTransformer.pdiparams
    ```
    config 给出模型推理设置 ```cfg.INFERENCE```
    输入视频为 ```BrushingTeeth.avi``` 主要输出结果如下:
    ```
    Current video file: BrushingTeeth.avi
	top-1 class: 19
	top-1 score: 0.9959074258804321
    ```


## 6. LICENSE
本项目的发布受[Apache 2.0 license](https://github.com/zwtu/TokShift-Transformer-Paddle/blob/main/LICENSE)许可认证。

## 7. 致谢
感谢百度飞浆团队提供的算力支持！