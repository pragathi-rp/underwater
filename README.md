# Underwater Image Enhancement with Reinforcement Learning 

#### Prerequisites

1.  CUDA 11.0
2.  Python 3.7
3.  TensorFlow 2.4

#### Compilation
Install all the python dependencies using pip:
        
        pip install -r requirements.txt

#### Data Preparation
Prepare the dataset according to [https://li-chongyi.github.io/proj_benchmark.html](https://li-chongyi.github.io/proj_benchmark.html) and put the data into the corresponding folder as follows:

        RL
        └── data
            ├── train
            │   ├── target
            │   └── raw
            └── test
                 ├── target
                 └── raw

#### Training
1. Clone the repo
2. Download the VGG-pretrained model from [VGG in Tensorflow](https://github.com/jcheng1602/tensorflow-vgg)
3. Put the training data to corresponding folders
4. CUDA_VISIBLE_DEVICES=1 python3 main.py --prefix train_model
5. Find checkpoints in the ./checkpoints/ 

#### Implementation
1. Clone the repo
2. Change the default value of --test to True and the default value of --model_path to $model path in ./checkpoints
3. Download the checkpoint from [Baidu Cloud](https://pan.baidu.com/s/1NLVFlfivIm-tyAJut73vQw) (Password: 1314)
4. Put the data to corresponding folders (target images are only used for scoring and do not participate in the implementation process)
5. CUDA_VISIBLE_DEVICES=1 python3 main.py --test --prefix test_model 
6. Find enhancement results in the ./test/test_model/step_0000000000

#### Results
![Image text](https://gitee.com/sunshixin_upc/underwater-image-enhancement-with-reinforcement-learning/blob/master/Experimental%20results/Testing%20set/86.png)