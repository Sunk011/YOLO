
import os
import sys
from datetime import datetime
def print_log(s, content_color = 'green'):
    '''
    PrintColor = <black> <red> <green> <yellow> <blue> <amaranth>  <ultramarine> <white> \\
    PrintStyle = <default> <highlight> <underline> <flicker> <inverse> <invisible>
    '''
    # 直接转成字符串  如果输入 颜色 配置选项不在选项中
    if content_color not in ['black','red','green','yellow','blue','amaranth','ultramarine','white','']:
        s = s + '\033[0;31m x_x \033[0m' +  str(content_color)
        content_color = 'green'
    PrintColor = {'black': 30,'red': 31,'green': 32,'yellow': 33,'blue': 34,'amaranth': 35,'ultramarine': 36,'white': 37}
    PrintStyle = {'default': 0,'highlight': 1,'underline': 4,'flicker': 5,'inverse': 7,'invisible': 8}

    time_style = PrintStyle['default']
    content_style = PrintStyle['default']
    time_color = PrintColor['blue']
    content_color = PrintColor[content_color]

    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print (log)

import argparse
# 添加命令行参数解析
parser = argparse.ArgumentParser(description='YOLO Training Script')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume training')
# model_yaml
parser.add_argument('--model_yaml', type=str, default='', help='Path to the model YAML configuration file')
# batch 参数
parser.add_argument('--batch', type=int, default=16, help='Batch size for training (default: 16)')
# amp 参数
parser.add_argument('--amp_false', action='store_false', help='Disable Automatic Mixed Precision training (default: True)')
# device 参数
parser.add_argument('--device', type=str, default=None, help='Training device(s) (e.g., "0,1" or "6,7"). If not specified, uses default device.')
#ultralytics_type
parser.add_argument('--ultralytics_type', type=str, default='ProModel-20250612', help='ultralytics_path_type <ProModel-20250601, ProModel-20250612, ultralytics, SPAR, FCM, yolov13>')
#YOLO_type
parser.add_argument('--YOLO_type', type=str, default='YOLO', help='YOLO_type <YOLO, SPARYOLO>')
# use_deterministic_false
parser.add_argument('--use_deterministic_false', action='store_true', help='Disable deterministic mode (default: True)')
args = parser.parse_args()


if args.ultralytics_type == 'ProModel-20250601':
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ProModelZoo/ultralytics-yolo11-20250601'
elif args.ultralytics_type == 'ProModel-20250612':
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ProModelZoo/ultralytics-yolo11-20250612'
elif args.ultralytics_type == 'ultralytics': 
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ultralytics'
elif args.ultralytics_type == 'SPAR':
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/Analogical-Reasoning/Analogical'
elif args.ultralytics_type == 'FCM':
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/tmp/FCM'
elif args.ultralytics_type == 'yolov13':
    Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ProModelZoo/yolov13'
else:
    print_log(f'======================>>> 未知的 opt.ultralytics_type: {args.ultralytics_type} <<<======================', 'red')
    # 打印ultralytics_type可选项
    print_log('======================>>> 可选项为: ProModel-20250601, ultralytics, SPAR, FCM <<<======================', 'red')
    exit()

# ======================= 设置环境变量 =======================

# Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ultralytics'
# Ultralytics_path = '/userhome/lhf/Codes/3rdparty/Analogical-Reasoning/Analogical'

# Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ProModelZoo/ultralytics-yolo11-20250601'
# Ultralytics_path = '/userhome/lhf/Codes/3rdparty/ProModelZoo/ultralytics-yolo11-20250612'

## 给多卡并行时候 DDP 临时脚本用的
os.environ['PYTHONPATH'] = os.pathsep.join([
    os.environ.get('PYTHONPATH', ''),
    Ultralytics_path
])

## 给当前代码 import ultralytics 用的
sys.path.append(Ultralytics_path)


# for Datasets
# os.environ['VisDrone2019_LABELS_PATH'] = '/userhome/lhf/datasets/VisDrone/labels'


# os.environ['polaris_env_Flag'] = 'True'
if args.use_deterministic_false:
    os.environ['polaris_env_Flag_use_deterministic'] = 'False'  # 设置为 False 以避免使用 deterministic 模式 默认为True
    print_log(">>> polaris_env_Flag_use_deterministic set to False, avoiding deterministic mode <<<", 'yellow')
# os.environ['polaris_env_not_use_rectangle'] = 'True' # 强制使用 正方形 不使用 rect  默认为 False


# 使用变量的地方这样设置  os.environ.get('polaris_env_not_use_rectangle', 'False').lower() == 'true'
# ======================= 设置环境变量 =======================
if args.YOLO_type == 'SPARYOLO':
    print(f"Environment variable: {os.environ.get('polaris_env_Flag_use_deterministic', 'Not Set')}")

    from ultralytics import SPARYOLO
    print_log(f"======================>>> Using SPARYOLO <<<======================", 'green')
else:
    print(f"Environment variable: {os.environ.get('polaris_env_Flag_use_deterministic', 'Not Set')}")

    from ultralytics import YOLO
    print_log(f"======================>>> Using YOLO <<<======================", 'green')
# from ultralytics import YOLO

def check_model_path(model_path,model_type='best'):
    """
    检查模型路径是否有效。如果路径以 .pt 结尾，则直接返回。
    否则，检查 {model_path}/weights/best.pt 是否存在。
    """
    # 如果路径以 .pt 结尾，直接返回
    if model_path.endswith('.pt'):
        if os.path.exists(model_path):
            print_log(f">>> 模型权重路径存在: {model_path} <<<+")
            return model_path
        else:
            print_log(f">>> 模型权重路径不存在: {model_path} <<<+", 'yellow')
            exit()

    # 如果路径不以 .pt 结尾，检查 {model_path}/weights/best.pt
    best_pt_path = os.path.join(model_path, "weights", f"{model_type}.pt")
    if os.path.exists(best_pt_path):
        print_log(f">>> 发现模型权重路径: {best_pt_path} <<<+")
        return best_pt_path
    else:
        print_log(f">>> 模型权重路径不存在: {best_pt_path} <<<+", 'yellow')
        exit()

if __name__ == '__main__':
    

    # Load a model
    # model = YOLO("yolov8m.yaml")  # build a new model from YAML
    # YOLOm_Pretrained = "/userhome/lhf/Codes/PreTrainWeights/yolo11m.pt"
    if args.resume:
        model_path = check_model_path(args.resume,model_type='last')
        print(f"Resuming training from {model_path}")
        if args.YOLO_type == 'SPARYOLO':
            print_log(f"======================>>> Using SPARYOLO with model_path: {model_path} <<<======================", 'green')
            model = SPARYOLO(model_path)  # load a pretrained model (recommended for training)
        else:
            model = YOLO(model_path)  # load a pretrained model (recommended for training)
        resume_flag = True
    else:
        if args.model_yaml is None:
            print_log(">>> No model YAML provided, using default YOLOv8m configuration <<<", 'yellow')
            # model_yaml = "yolov8m.yaml"
            exit(1)
        model_yaml = str(args.model_yaml)
        if args.YOLO_type == 'SPARYOLO':
            print_log(f"======================>>> Using SPARYOLO with model_yaml: {model_yaml} <<<======================", 'green')
            model = SPARYOLO(model_yaml)  # load a pretrained model (recommended for training)
        else:
            print_log(f"======================>>> Using YOLO with model_yaml: {model_yaml} <<<======================", 'green')
            model = YOLO(model_yaml)  # load a pretrained model (recommended for training)
        resume_flag = False
    # model = SPARYOLO("/userhome/lhf/Codes/3rdparty/Analogical-Reasoning/Analogical/ultralytics/cfg/models/11/yolo11m-spar.yaml")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # 根据debug参数设置训练配置
    if args.debug:
        train_name = "Debug"
        default_train_device = "4,5"
        imgsz = 1024
        print_log(" ==============================================> Running in DEBUG mode")
    else:
        # 设置分辨率
        imgsz = 1024
        # 从 model_yaml 路径中提取文件名（不含扩展名）
        model_yaml_filename = os.path.splitext(os.path.basename(args.model_yaml))[0]
        # 根据 yaml 文件名和分辨率生成 train_name
        train_name = f"{model_yaml_filename}-{imgsz}"
        # 默认训练设备
        default_train_device = "6,7"
        print_log(f" ==============================================> Running in NORMAL mode with train_name: {train_name}")

    # 如果命令行指定了设备，就使用指定的设备，否则使用默认设备
    train_device = args.device if args.device is not None else default_train_device
    print_log(f" ==============================================> Using training device: {train_device}")

    # Train the model
    model.train(
        data="/userhome/lhf/Github/Datasets/VisDrone/visdrone.yaml", 
        epochs=100, 
        imgsz=imgsz, 
        batch=args.batch,
        #  ======== for Debugging
        # name="Debug-1",
        # device="0,1",

        # set ther esume=True,  # resume most recent training
        resume=resume_flag,

        #  ======== for multi-GPU training
        exist_ok=False,
        name=train_name,
        device=train_device,
        project="/userhome/lhf/Codes/WorkSpace/VisDrone/noPre/results",

        amp=args.amp_false,  # Automatic Mixed Precision training (default: True, use --amp to disable)

        ## 参考 SPARYOLO 的 train.py的训练超参数修改，期望对齐参数
        optimizer = 'AdamW',
        conf=0.2,
        iou=0.5,
        lr0=0.001,
        mixup=0.4,
        crop_fraction=1.0,
        )