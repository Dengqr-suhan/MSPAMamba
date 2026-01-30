## Potsdam数据集模型指令
### CMTFNet模型

python train_supervision.py --config_path config/potsdam/CMTFNet_config.py
python potsdam_test.py --config_path config/potsdam/CMTFNet_config.py --output_path fig_results/potsdam/cmtfnet --rgb -t 'd4'
F1_ImSurf:0.9203126174324374, IOU_ImSurf:0.8523880451801501
F1_Building:0.968467731183468, IOU_Building:0.9388632430225208
F1_LowVeg:0.8820183132914099, IOU_LowVeg:0.7889380691808365
F1_Tree:0.8980579523666253, IOU_Tree:0.8149774793469144
F1_Car:0.9544213810655992, IOU_Car:0.912816467152222
F1_Clutter:0.5782723118399419, IOU_Clutter:0.40673915029981467
F1:0.924655599067908, mIOU:0.8615966607765287, OA:0.9058467970201333

### UnetFormer模型

python train_supervision.py --config_path config/potsdam/unetformer.py 
python potsdam_test.py --config_path config/potsdam/unetformer.py --output_path fig_results/potsdam/unetformer --rgb -t 'd4'
F1_ImSurf:0.9145832014993744, IOU_ImSurf:0.842610140881146
F1_Building:0.9659705622195968, IOU_Building:0.9341809110319932
F1_LowVeg:0.8740728672395446, IOU_LowVeg:0.7763138855145667
F1_Tree:0.8904260053056879, IOU_Tree:0.8024935782232355
F1_Car:0.9430132221436016, IOU_Car:0.892171256915873
F1_Clutter:0.5129217630893342, IOU_Clutter:0.34491915109651844
F1:0.9176131716815611, mIOU:0.849553954513363, OA:0.8987039982380293
### RS3Mamba模型
python train_supervision.py --config_path config/potsdam/RS3Mamba_config.py 
python potsdam_test.py --config_path config/potsdam/RS3Mamba_config.py --output_path fig_results/potsdam/rs3mamba  --rgb -t 'd4'


F1_ImSurf:0.9169044441484814, IOU_ImSurf:0.846559141707142
F1_Building:0.9688437850248457, IOU_Building:0.9395703298439508
F1_LowVeg:0.8797326625369309, IOU_LowVeg:0.7852881478533086
F1_Tree:0.895383805676116, IOU_Tree:0.8105836310177985
F1_Car:0.954388567761527, IOU_Car:0.9127564392810305
F1_Clutter:0.4957323296807461, IOU_Clutter:0.32955061088000104
F1:0.9230506530295802, mIOU:0.858951537940646, OA:0.9020648153571772
images writing spends: 16.443620204925537 s
### PPMamba模型

python train_supervision.py --config_path config/potsdam/ppmamba.py 
python potsdam_test.py --config_path config/potsdam/ppmamba.py --output_path fig_results/potsdam/ppmamba  --rgb -t 'd4'


F1_ImSurf:0.922917200104351, IOU_ImSurf:0.8568674573521793
F1_Building:0.9656711005491648, IOU_Building:0.9336209217995132
F1_LowVeg:0.8790009962404365, IOU_LowVeg:0.7841229058121164
F1_Tree:0.8993490480629287, IOU_Tree:0.8171065009121512
F1_Car:0.9560422404439565, IOU_Car:0.915786325349529
F1_Clutter:0.5527048889294027, IOU_Clutter:0.38188817519085955
F1:0.9245961170801674, mIOU:0.8615008222450978, OA:0.9056112149072325



### unetMamba模型
python train_supervision.py --config_path config/potsdam/UnetMamba_config.py
python potsdam_test.py --config_path config/potsdam/UnetMamba_config.py --output_path fig_results/potsdam/UnetMamba --rgb -t 'd4'
F1_ImSurf:0.9201030692975406, IOU_ImSurf:0.8520285993395916
F1_Building:0.968909144903903, IOU_Building:0.9396932773820414
F1_LowVeg:0.8685825408434623, IOU_LowVeg:0.7676941290007873
F1_Tree:0.891131925787972, IOU_Tree:0.8036410701256943
F1_Car:0.9576901032804529, IOU_Car:0.9188151300247486
F1_Clutter:0.5044763437896365, IOU_Clutter:0.3373242152972508
F1:0.9212833568226662, mIOU:0.8563744411745727, OA:0.8981520198525655
images writing spends: 16.223366498947144 s
### MSPAMamba模型
python train_supervision.py --config_path config/potsdam/MSPAMamba_config.py
python potsdam_test.py --config_path config/potsdam/MSPAMamba_config.py --output_path fig_results/potsdam/MSPAMamba --rgb -t 'd4'

F1_ImSurf:0.9248100394984591, IOU_ImSurf:0.8601364163287624
F1_Building:0.9723333038164423, IOU_Building:0.9461562853281061
F1_LowVeg:0.886939058310595, IOU_LowVeg:0.796846807834617
F1_Tree:0.9023610763329064, IOU_Tree:0.822092818390783
F1_Car:0.9626149174687078, IOU_Car:0.9279243876534834
F1_Clutter:0.5367797172992917, IOU_Clutter:0.36684819343027536
F1:0.9298116790854222, mIOU:0.8706313431071504, OA:0.9095940576080662




python train_supervision.py --config_path config/potsdam/MSPAMamba_ablation1_config.py
python potsdam_test.py --config_path config/potsdam/MSPAMamba_ablation1_config.py --output_path fig_results/potsdam/MSPAMamba_ablation1 -t 'd4'
F1_ImSurf:0.9247805315287614, IOU_ImSurf:0.8600853673563285
F1_Building:0.9665595617467819, IOU_Building:0.9352832790059171
F1_LowVeg:0.8858541361354856, IOU_LowVeg:0.7950970917423875
F1_Tree:0.9002470925259952, IOU_Tree:0.8185903273433943
F1_Car:0.9561313737283387, IOU_Car:0.9159499094663954
F1_Clutter:0.6163341331863204, IOU_Clutter:0.44543567053917504
F1:0.9267145391330726, mIOU:0.8650011949828844, OA:0.9093755740048056
images writing spends: 15.296764135360718 s


python train_supervision.py --config_path config/potsdam/MSPAMamba_ablation2_config.py
python potsdam_test.py --config_path config/potsdam/MSPAMamba_ablation2_config.py --output_path fig_results/potsdam/MSPAMamba_ablation2 -t 'd4'

F1_ImSurf:0.9234303184255315, IOU_ImSurf:0.8577524838661879
F1_Building:0.9691401463487642, IOU_Building:0.9401279358354439
F1_LowVeg:0.8839180350251764, IOU_LowVeg:0.7919830825732551
F1_Tree:0.8993451302564097, IOU_Tree:0.8171000328794458
F1_Car:0.959101439205771, IOU_Car:0.92141681747927
F1_Clutter:0.5407143685672667, IOU_Clutter:0.3705336069377939
F1:0.9269870138523306, mIOU:0.8656760705267205, OA:0.9070675419953856

python train_supervision.py --config_path config/potsdam/MSPAMamba_ablation3_config.py
python potsdam_test.py --config_path config/potsdam/MSPAMamba_ablation3_config.py --output_path fig_results/potsdam/MSPAMamba_ablation3 -t 'd4'

F1_ImSurf:0.9258928039612513, IOU_ImSurf:0.8620115453801033
F1_Building:0.9705807153507887, IOU_Building:0.9428429502187998
F1_LowVeg:0.882957423228562, IOU_LowVeg:0.7904420490224761
F1_Tree:0.8982118198015959, IOU_Tree:0.8152309454253276
F1_Car:0.9629489918916362, IOU_Car:0.9285454470056457
F1_Clutter:0.5554368737836238, IOU_Clutter:0.3845016280032244
F1:0.9281183508467666, mIOU:0.8678145874104704, OA:0.9082404263889031

python train_supervision.py --config_path config/potsdam/MSPAMamba_baseline_config.py
python potsdam_test.py --config_path config/potsdam/MSPAMamba_baseline_config.py --output_path fig_results/potsdam/MSPAMamba_baseline -t 'd4'
F1_ImSurf:0.9216625207587689, IOU_ImSurf:0.8547069340549062
F1_Building:0.9676811643095427, IOU_Building:0.9373859420692615
F1_LowVeg:0.8804593326933438, IOU_LowVeg:0.7864469406113811
F1_Tree:0.8953406422448783, IOU_Tree:0.8105128843197248
F1_Car:0.9570613425721461, IOU_Car:0.9176583260730766
F1_Clutter:0.5868750650075751, IOU_Clutter:0.4153030283983497
F1:0.924441000515736, mIOU:0.8613422054256701, OA:0.9057277458478289
images writing spends: 12.605875015258789 s






python model_metrics_calculator.py --config_path config/potsdam/MSPAMamba_config.py
python model_metrics_calculator.py --config_path config/potsdam/RS3Mamba_config.py
python model_metrics_calculator.py --config_path config/potsdam/PPMamba.py
python model_metrics_calculator.py --config_path config/potsdam/UnetMamba_config.py
python model_metrics_calculator.py --config_path config/potsdam/CMTFNet_config.py
python model_metrics_calculator.py --config_path config/potsdam/HMAFNet_config.py

python model_metrics_calculator.py --config_path config/potsdam/MSPAMamba_ablation1_config.py
python model_metrics_calculator.py --config_path config/potsdam/MSPAMamba_ablation2_config.py
python model_metrics_calculator.py --config_path config/potsdam/MSPAMamba_ablation3_config.py
python model_metrics_calculator.py --config_path config/potsdam/MSPAMamba_baseline_config.py

