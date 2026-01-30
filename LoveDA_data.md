## LoveDA数据集模型指令

### UNetFormer模型

python train_supervision.py --config_path config/loveda/unetformer.py
python loveda_test.py --config_path config/loveda/unetformer.py --output_path fig_results/loveda/unetformer --rgb -t 'd4'

### RS3Mamba模型

python train_supervision.py --config_path config/loveda/rs3mamba.py
python loveda_test.py --config_path config/loveda/rs3mamba.py --output_path fig_results/loveda/rs3mamba --rgb -t 'd4'

### MSPAMamba模型

python train_supervision.py --config_path config/loveda/mspamamba.py
python loveda_test.py --config_path config/loveda/mspamamba.py --output_path fig_results/loveda/mspamamba --val --rgb -t 'd4'

### PPMamba模型

python train_supervision.py --config_path config/loveda/ppmamba.py
python loveda_test.py --config_path config/loveda/ppmamba.py --output_path fig_results/loveda/ppmamba --rgb -t 'd4'

### UNetMamba模型

python train_supervision.py --config_path config/loveda/unetmamba.py
python loveda_test.py --config_path config/loveda/unetmamba.py --output_path fig_results/loveda/unetmamba --rgb -t 'd4'

### CMTFNet模型

python train_supervision.py --config_path config/loveda/cmtfnet.py
python loveda_test.py --config_path config/loveda/cmtfnet.py --output_path fig_results/loveda/cmtfnet --rgb --val -t 'd4'