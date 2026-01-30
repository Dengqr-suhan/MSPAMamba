
## Vaihingen数据集模型指令

### CMTFNet模型

python train_supervision.py --config_path config/vaihingen/CMTFNet_config.py
python vaihingen_test.py --config_path config/vaihingen/CMTFNet_config.py --output_path fig_results/vaihingen/cmtfnet --rgb

F1_ImSurf:0.969169539604001, IOU_ImSurf:0.9401832569360525
F1_Building:0.9619528008509541, IOU_Building:0.9266946644040163
F1_LowVeg:0.8151695920578982, IOU_LowVeg:0.6880052930729073
F1_Tree:0.9117493207445666, IOU_Tree:0.837811855415862
F1_Car:0.8655482424449437, IOU_Car:0.7629661082374739
F1_Clutter:0.9878365846388236, IOU_Clutter:0.9759655107533479
F1:0.9047178991404727, mIOU:0.8311322356132624, OA:0.9407459488220486
images writing spends: 3.370598554611206 s
### UnetFormer模型

python train_supervision.py --config_path config/vaihingen/unetformer.py 
python vaihingen_test.py --config_path config/vaihingen/unetformer.py --output_path fig_results/vaihingen/unetformer  --rgb

F1_ImSurf:0.9675207616107833, IOU_ImSurf:0.9370849559359895
F1_Building:0.9590794526392095, IOU_Building:0.9213762328651445
F1_LowVeg:0.8102440115172629, IOU_LowVeg:0.6810169642857142
F1_Tree:0.9087212759920272, IOU_Tree:0.8327123547818643
F1_Car:0.8660683168615033, IOU_Car:0.7637746874347834
F1_Clutter:0.9726844328891777, IOU_Clutter:0.9468214675503392
F1:0.9023267637241572, mIOU:0.8271930390606992, OA:0.9384295837785462

### RS3Mamba模型

python train_supervision.py --config_path config/vaihingen/RS3Mamba_config.py
python vaihingen_test.py --config_path config/vaihingen/RS3Mamba_config.py --output_path fig_results/vaihingen/rs3mamba  --rgb

F1_ImSurf:0.9681676830577444, IOU_ImSurf:0.9382994379618039
F1_Building:0.9604938522637971, IOU_Building:0.9239905452752964
F1_LowVeg:0.8208750521786893, IOU_LowVeg:0.696173084706107
F1_Tree:0.9158522509123193, IOU_Tree:0.8447670086323718
F1_Car:0.8683787376444021, IOU_Car:0.7673757700847484
F1_Clutter:0.9705547475586952, IOU_Clutter:0.9427939419381922
F1:0.9067535152113905, mIOU:0.8341211693320656, OA:0.9413152060469985


### PPMamba模型
python train_supervision.py --config_path config/vaihingen/ppmamba.py
python vaihingen_test.py --config_path config/vaihingen/ppmamba.py --output_path fig_results/vaihingen/ppmamba  --rgb
F1_ImSurf:0.9691141012870897, IOU_ImSurf:0.9400789190123326
F1_Building:0.9593084668437114, IOU_Building:0.9217990502279265
F1_LowVeg:0.8218691461131852, IOU_LowVeg:0.6976042970114283
F1_Tree:0.9095371095117968, IOU_Tree:0.8340835047624542
F1_Car:0.8859878538760426, IOU_Car:0.7953125618591393
F1_Clutter:0.9664010860255021, IOU_Clutter:0.9349865532553631
F1:0.9091633355263651, mIOU:0.8377756665746562, OA:0.9403321053313646


### unetMamba模型
python train_supervision.py --config_path config/vaihingen/UnetMamba_config.py
python vaihingen_test.py --config_path config/vaihingen/UnetMamba_config.py --output_path fig_results/vaihingen/unetMamba  --rgb
F1_ImSurf:0.9713519436378114, IOU_ImSurf:0.9442995956002631
F1_Building:0.9644068717449705, IOU_Building:0.9312604008583877
F1_LowVeg:0.8144552585138354, IOU_LowVeg:0.6869882088910941
F1_Tree:0.9065337930011043, IOU_Tree:0.8290460072736567
F1_Car:0.9054433801831173, IOU_Car:0.8272238857178503
F1_Clutter:0.9816471667813719, IOU_Clutter:0.9639558459112413
F1:0.9124382494161678, mIOU:0.8437636196682504, OA:0.9419891072576998


### MSPAMamba模型
python train_supervision.py --config_path config/vaihingen/MSPAMamba_config.py
python vaihingen_test.py --config_path config/vaihingen/MSPAMamba_config.py --output_path fig_results/vaihingen/MSPAMamba  --rgb
F1_ImSurf:0.9711950674355523, IOU_ImSurf:0.9440031211890727
F1_Building:0.9636999168420985, IOU_Building:0.9299429118112493
F1_LowVeg:0.8230694430970831, IOU_LowVeg:0.6993356050360213
F1_Tree:0.9122144197106509, IOU_Tree:0.8385976393141775
F1_Car:0.9019312788244503, IOU_Car:0.8213796290079893
F1_Clutter:0.9903377440931023, IOU_Clutter:0.9808604197089275
F1:0.914422025181967, mIOU:0.846651781271702, OA:0.9436434590107345











python train_supervision.py --config_path config/vaihingen/MSPAMamba_ablation1_config.py
python vaihingen_test.py --config_path config/vaihingen/MSPAMamba_ablation1_config.py --output_path fig_results/vaihingen/MSPAMamba_ablation1

F1_ImSurf:0.9682388973843671, IOU_ImSurf:0.9384332234756382
F1_Building:0.9619488937490632, IOU_Building:0.926687412552618
F1_LowVeg:0.8259546418547252, IOU_LowVeg:0.7035116966516066
F1_Tree:0.9118767924809544, IOU_Tree:0.8380271518701101
F1_Car:0.8894373298923239, IOU_Car:0.8008889131903626
F1_Clutter:0.9765395528929843, IOU_Clutter:0.9541546580069008
F1:0.9114913110722869, mIOU:0.8415096795480672, OA:0.9416998597717583



python train_supervision.py --config_path config/vaihingen/MSPAMamba_ablation2_config.py
python vaihingen_test.py --config_path config/vaihingen/MSPAMamba_ablation2_config.py --output_path fig_results/vaihingen/MSPAMamba_ablation2

F1_ImSurf:0.9709575196301528, IOU_ImSurf:0.9435543606335686
F1_Building:0.9631526250811161, IOU_Building:0.9289242065703901
F1_LowVeg:0.8232491814701342, IOU_LowVeg:0.6995951636546407
F1_Tree:0.9103029131068322, IOU_Tree:0.8353724388694055
F1_Car:0.886155703023493, IOU_Car:0.7955831038763075
F1_Clutter:0.9406529479878436, IOU_Clutter:0.8879554119692301
F1:0.9107635884623455, mIOU:0.8406058547208624, OA:0.9424464893097495
images writing spends: 2.7628886699676514 s

python train_supervision.py --config_path config/vaihingen/MSPAMamba_ablation3_config.py
python vaihingen_test.py --config_path config/vaihingen/MSPAMamba_ablation3_config.py --output_path fig_results/vaihingen/MSPAMamba_ablation3
F1_ImSurf:0.9682784885730672, IOU_ImSurf:0.9385076087382145
F1_Building:0.9581929724537293, IOU_Building:0.9197413216826974
F1_LowVeg:0.8262938499501928, IOU_LowVeg:0.7040040217179814
F1_Tree:0.9128403437485417, IOU_Tree:0.839656198148511
F1_Car:0.8863744582273042, IOU_Car:0.7959358195182487
F1_Clutter:0.9606909400446494, IOU_Clutter:0.9243553982738505
F1:0.910396022590567, mIOU:0.8395689939611307, OA:0.9408206458495045


python train_supervision.py --config_path config/vaihingen/MSPAMamba_baseline_config.py
python vaihingen_test.py --config_path config/vaihingen/MSPAMamba_baseline_config.py --output_path fig_results/vaihingen/MSPAMamba_baseline

F1_ImSurf:0.9689903560195092, IOU_ImSurf:0.9398460641730378
F1_Building:0.9621973028602279, IOU_Building:0.9271485856724829
F1_LowVeg:0.8156379356519737, IOU_LowVeg:0.6886727971154415
F1_Tree:0.9068878821312453, IOU_Tree:0.8296384856655036
F1_Car:0.8957189804300663, IOU_Car:0.8111331849015273
F1_Clutter:0.9815782328596935, IOU_Clutter:0.9638229116173856
F1:0.9098864914186044, mIOU:0.8392878235055985, OA:0.9403704571249848
images writing spends: 2.453521966934204 s



python model_metrics_calculator.py --config_path config/vaihingen/MSPAMamba_config.py
python model_metrics_calculator.py --config_path config/vaihingen/RS3Mamba_config.py
python model_metrics_calculator.py --config_path config/vaihingen/PPMamba.py
python model_metrics_calculator.py --config_path config/vaihingen/UnetMamba_config.py
python model_metrics_calculator.py --config_path config/vaihingen/CMTFNet_config.py
python model_metrics_calculator.py --config_path config/vaihingen/HMAFNet_config.py



python model_metrics_calculator.py --config_path config/vaihingen/MSPAMamba_ablation1_config.py
python model_metrics_calculator.py --config_path config/vaihingen/MSPAMamba_ablation2_config.py
python model_metrics_calculator.py --config_path config/vaihingen/MSPAMamba_ablation3_config.py
python model_metrics_calculator.py --config_path config/vaihingen/MSPAMamba_baseline_config.py