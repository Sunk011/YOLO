import os
import cv2
import xml.etree.ElementTree as ET
# from xml.etree import ElementTree
def print_color(data, color='green'):
    """
    颜色样式打印输出功能
    :param data: 打印内容
    :param color: 指定颜色，默认为绿色('green')
    :return:
    """
    # 定义颜色名称到ANSI代码的映射
    color_map = {
        'black': '30',  # 黑色
        'red': '31',    # 红色
        'green': '32',  # 绿色
        'yellow': '33', # 黄色
        'blue': '34',   # 蓝色
        'magenta': '35',# 品红
        'cyan': '36',   # 青色
        'white': '37'   # 白色
    }
    
    # 获取颜色代码，若输入的颜色不在映射中，则默认使用绿色
    ansi_color = color_map.get(color, '32')
    
    print(f"\033[1;{ansi_color}m{data}\033[0m")


class XMLLabelSummarizer:
    '''
    统计xml文件夹及其子文件夹中所有xml的标签
    '''

    def __init__(self, folder_path):
        self.folder_path = folder_path  # the folder_path of the xml folder
        self.name_count = {}
        self.class_names = set()

    def parse_xml_file(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        labels = set()  # Use a set to store labels and automatically remove duplicates
        for object_elem in root.findall('object'):
            label = object_elem.find('name').text
            labels.add(label)
        self._count_names(root)
        return labels

    def summarize_labels(self,folder_path):
        labels = set()  # Use a set to store labels and automatically remove duplicates
        for root_dir, dirs, files in os.walk(folder_path):
            print_color(f"===>  summarize files from path {root_dir} ",'green')
            if files:
                print(f"===>  Find {len(files)} files in path {root_dir} ")
                for file in files:
                    if file.endswith('.xml'):
                        xml_file = os.path.join(root_dir, file)
                        labels |= self.parse_xml_file(xml_file)  # Use the bitwise OR operator to merge multiple sets
            else:
                print_color(f"===>  No xml found in path {root_dir}",color='red')
        self.class_names = labels
        return labels

    def print_all_labels(self):
        all_labels = self.summarize_labels(self.folder_path)
        print_color(f"======> all classes set in {self.folder_path} <======",'green')
        for label in all_labels:
            print(label)
        # for name, count in self.name_count.items():
        #     print(f"{name}: {count}")
        # 找到最长的键的长度，以便对齐
        max_key_length = max(len(key) for key in self.name_count)
        # 打印标题
        print(f"{'Item':<{max_key_length}} | {'Count'}")
        # 打印分隔线
        print("-" * (max_key_length + 9))
        for key, value in self.name_count.items():
            print(f"{key:<{max_key_length}} | {value}")
        print_color(f"======> all classes set in {self.folder_path} <======",'green')

    def _count_names(self, element):
        if element.tag == 'object':
            name_element = element.find('name')
            if name_element is not None:
                name = name_element.text.strip()
                self.name_count[name] = self.name_count.get(name, 0) + 1
        for child in element:
            self._count_names(child)

    def get_class_list(self):
        sorted_keys = sorted(self.name_count, key=self.name_count.get, reverse=True)
        return list(sorted_keys)

class XMLtoTXTConverter:
    '''
    input_dir # 存放的xml文件地址
    out_dir # 转换为txt后保存的地址
    txt_write_mode  'w' 表示覆盖 'a' 表示追加

    将xml标签转换为yolo的txt格式
    TODO 1 : 
        如果 class_list 已经存在 则根据 现有的list确定ID 传入参数为 class_names.txt 默认为空 如果为空则随机生成然后给出txt文件
    '''

    def __init__(self, input_dir, out_dir, class_dir=None, class_list=None, class_map_dist=None, txt_write_mode='w'):
        self.input_dir = input_dir  # 存放的xml文件地址
        self.out_dir = out_dir  # 转换为txt后保存的地址
        self.class_dir = class_dir
        self.class_list = class_list
        self.class_map_dist = class_map_dist
        self.txt_write_mode = txt_write_mode  # 追加 or 覆盖
        if not (class_dir or class_list or class_map_dist):
            print_color(f"class_dir or class_list or class_map_dist must be provided", color='red')
            exit()
        # self.class_list = ['person', 'car', 'truck', 'bus', 'van', 'motor', 'tricycle', 'tractor', 'camping car',
                        #    'awning-tricycle', 'bicycle', 'trailer']  # xml的类别
        # self.class_list = ['airplane','airport_tower','bridge','vehicles','ship','missile_vehicle','missile_defense_site','radar_vehicle','robomaster']
        # '''
        #     airplane
        #     airport_tower
        #     bridge
        #     vehicles
        #     ship
        #     missile_vehicle
        #     missile_defense_site
        #     radar_vehicle
        #     robomaster

        # '''

        os.makedirs(self.out_dir, exist_ok=True)

    def convert(self):
        file_name_list,file_path_list = self._get_file_list()
        # self._get_class(filelist)
        self._convert_xml_to_txt(file_path_list)
        if self.class_dir: self._create_class_file()

    # def convert_for_all_subpath(self):
        
    def _get_file_list(self):
        file_name_list = []
        file_path_list = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.xml'):
                    file_name = os.path.splitext(file)[0]
                    file_path = os.path.join(root,file)
                    file_path_list.append(file_path)
                    file_name_list.append(file_name)
        return file_name_list,file_path_list

    def _convert_xml_to_txt(self, file_path_list):
        for file_path in file_path_list:
            # file_path = os.path.join(self.input_dir, i + ".xml")
            outfile = open(file_path, encoding='UTF-8')
            filetree = ET.parse(outfile)
            outfile.close()
            root = filetree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            imgshape = (width, height)

            txtresult = ''
            for obj in root.findall('object'):
                obj_name = obj.find('name').text
                # TODO 指定表现映射后 修改id映射
                if self.class_map_dist and self.class_list:
                    if obj_name not in self.class_map_dist.keys():
                        continue
                    new_class_name = self.class_map_dist[obj_name]
                    obj_id = self.class_list.index(new_class_name)
                elif self.class_list:
                    if obj_name not in self.class_list:
                        continue
                    obj_id = self.class_list.index(obj_name)
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymin = float(bbox.find('ymin').text)
                    ymax = float(bbox.find('ymax').text)
                    bbox_coor = (xmin, xmax, ymin, ymax)

                    x, y, w, h = self._convert_coordinate(imgshape, bbox_coor)
                    txt = '{} {} {} {} {}\n'.format(obj_id, x, y, w, h)
                    txtresult = txtresult + txt
            txt_file_path = os.path.basename(file_path).replace('.xml','.txt')
            txt_file_path = os.path.join(self.out_dir, txt_file_path)
            f = open(txt_file_path, self.txt_write_mode)
            f.write(txtresult)
            f.close()
            print(f"convert {file_path} to {txt_file_path} done")

    def _convert_coordinate(self, imgshape, bbox):
        xmin, xmax, ymin, ymax = bbox
        width = imgshape[0]
        height = imgshape[1]
        dw = 1. / width
        dh = 1. / height
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw
        y = y * dh
        w = w * dw
        h = h * dh
        return x, y, w, h

    def _create_class_file(self):
        class_file_path = os.path.join(self.class_dir, "classes.txt")
        f = open(class_file_path, 'a')
        class_result = ''
        for i in self.class_list:
            class_result = class_result + i + "\n"
        f.write(class_result)
        f.close()
        


class TxtToXmlConverter:
    '''
    将yolo的txt标签转换为xml格式
    '''
    def __init__(self, classname_path, txt_path, img_path, xml_path):
        self.classname_path = classname_path  # 存放classes文.txt的地址，需要准备好的，里面写有标签
        self.txt_path = txt_path   # 待转换txt文件的地址
        self.img_path = img_path   # 存放图片文件的地址
        self.xml_path = xml_path   # 存放xml文件的地址
        # if path not exists, create it
        if not os.path.exists(self.xml_path):
            os.makedirs(self.xml_path)

    def check_empty_files(self):
        empty_files = []

        for file_name in os.listdir(self.txt_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.txt_path, file_name)
                if os.stat(file_path).st_size == 0:
                    empty_files.append(file_name)

        return empty_files

    def convert(self):
        # 1. Read the class labels from the classname file
        with open(self.classname_path, 'r') as f:
            classes = f.readlines()
            classes = [cls.strip('\n') for cls in classes]

        # 2. Find the txt label files
        files = os.listdir(self.txt_path)
        pre_img_name = ''

        # 3. Iterate over the files
        for i, name in enumerate(files):
            if name == '.DS_Store':
                continue

            txtFile = open(os.path.join(self.txt_path, name))
            txtList = txtFile.readlines()
            img_name = name.split(".")[0]
            # imgdir = os.path.join(self.img_path, img_name + ".jpg")
            '''
            TODO 2 通过 os.path.exists(file_path) 判断图片文件名
            文件不存在 跑出文件完整路径以便检查
            
            TODO 3 增加文件路径 检查机制 正反斜杠自动替换
            '''
            # imgdir = self.img_path + "/" + img_name + ".jpg"
            # check file if 
            img_exists_flag = False
            for img_shuffix in ['.jpg', '.png', '.jpeg']:
                imgdir = self.img_path + "/" + img_name + img_shuffix
                if os.path.exists(imgdir):
                    img_exists_flag = True
                    break
            if not img_exists_flag: 
                print(f" warning ... Unable to find image file: {img_name}")
                continue  
            pic = cv2.imread(imgdir)

            if pic is None:
                print(f"{img_name} image file exists, but unable to load image file, check authority or check if image file is normal ... ")
                continue

            Pheight = pic.shape[0]
            Pwidth = pic.shape[1]
            Pdepth = pic.shape[2]

            for txt_line_index,row in enumerate(txtList):
                oneline = row.strip().split(" ")

                if img_name != pre_img_name:
                    xml_file = open((os.path.join(self.xml_path, img_name + '.xml')), 'w')
                    xml_file.write('<annotation>\n')
                    xml_file.write('    <folder>images</folder>\n')
                    # TODO 参数不要写死
                    xml_file.write('    <filename>' + img_name + '.jpg' + '</filename>\n')
                    xml_file.write('    <path>' + str(imgdir) + '</path>\n')
                    # xml_file.write('    <imglab>Russia-Ukraine War</imglab>\n')
                    # TODO 参数不要写死
                    xml_file.write('    <source>\n')
                    xml_file.write('        <database>Unknown</database>\n')
                    xml_file.write('    </source>\n')
                    xml_file.write('    <size>\n')
                    xml_file.write('        <width>' + str(Pwidth) + '</width>\n')
                    xml_file.write('        <height>' + str(Pheight) + '</height>\n')
                    xml_file.write('        <depth>' + str(Pdepth) + '</depth>\n')
                    xml_file.write('    </size>\n')
                    #<segmented>0</segmented>
                    xml_file.write('    <segmented>0</segmented>\n')
                    xml_file.write('    <object>\n')
                    xml_file.write('        <name>' + classes[int(oneline[0])] + '</name>\n')
                    #<pose>Unspecified</pose>
                    xml_file.write('        <pose>Unspecified</pose> \n')
                    xml_file.write('        <truncated>0</truncated> \n')
                    xml_file.write('        <difficult>' + str(0) + '</difficult>\n')
                    xml_file.write('        <bndbox>\n')
                    xml_file.write('            <xmin>' + str(
                        int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                    xml_file.write('            <ymin>' + str(
                        int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                    xml_file.write('            <xmax>' + str(
                        int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                    xml_file.write('            <ymax>' + str(
                        int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                    xml_file.write('        </bndbox>\n')
                    xml_file.write('    </object>\n')
                    if txt_line_index == len(txtList)-1:
                        xml_file.write('</annotation>')
                    xml_file.close()
                    pre_img_name = img_name
                else:
                    xml_file = open((os.path.join(self.xml_path, img_name + '.xml')), 'a')
                    xml_file.write('    <object>\n')
                    xml_file.write('        <name>' + classes[int(oneline[0])] + '</name>\n')
                    #<pose>Unspecified</pose>
                    xml_file.write('        <pose>Unspecified</pose> \n')
                    xml_file.write('        <truncated>0</truncated> \n')
                    xml_file.write('        <difficult>' + str(0) + '</difficult>\n')
                    xml_file.write('        <bndbox>\n')
                    xml_file.write('            <xmin>' + str(
                        int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                    xml_file.write('            <ymin>' + str(
                        int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                    xml_file.write('            <xmax>' + str(
                        int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                    xml_file.write('            <ymax>' + str(
                        int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                    xml_file.write('        </bndbox>\n')
                    xml_file.write('    </object>\n')
                    if txt_line_index == len(txtList)-1:
                        xml_file.write('</annotation>')
                    xml_file.close()
                    # TODO 只加名字 不加坐标？？

class XmlNameModify:
    def __init__(self,xml_path):
        self.xml_path = xml_path
        ''' name_calss_dict 
            key 表示最终的label
            values 是要替换的labels
        '''
        self.name_calss_dict = {
            'Armored-car':['CM32','AFV'],
            'Tank':['M1A1','M1A2'],
            'Radar-car':['SA-15','SAM'],
            'Command-car':['LAV-AD'],
            'Launch-car ':['TianGong1','TianGong3'],
            'FieldWork':['FieldWor']
        }

    # def name2class(self):
    #     pass

    # def get_label_name(self,xml_file):
    #     tree = ET.parse(xml_file)
    #     root = tree.getroot()
    #     labels = set()  # Use a set to store labels and automatically remove duplicates
    #     for object_elem in root.findall('object'):
    #         label = object_elem.find('name').text
    #         labels.add(label)
    #     print(f'the labels of {xml_file} is {labels}')
    #     return labels

    def modify_self(self):
        '''
        没有指定名称修改对应关系的情况，修改类别名格式，例如 ”SAM 1“ 给为 ’SAM‘
        '''
        old_name = None
        new_name = None
        for root_path, dirs, files in os.walk(self.xml_path):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root_path, file)
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    for obj in root.iter('object'):
                        name = obj.find('name').text
                        # if name == 'FieldWor':
                        # print(f"===> find {file_path} name {name} ")
                        if " " in name:
                            old_name = name
                            new_name = str(name).split(' ')[0]
                            # Check if the 'name_old' element exists
                            name_old = obj.find('name_old')
                            if name_old is None:
                                # If 'name_old' element does not exist, create it
                                name_old = ET.SubElement(obj, 'name_old')
                            # Set 'name_old' text to the old name
                            name_old.text = name

                            obj.find('name').text = new_name
                            tree.write(file_path)
                            print(f"modify {file_path} name {old_name} to {new_name}")
                        
                        for key, value in self.name_calss_dict.items():
                            if name in value:
                                old_name = name
                                new_name = key
                                name_old = obj.find('name_old')
                                if name_old is None:name_old = ET.SubElement(obj, 'name_old')
                                name_old.text = old_name
                                obj.find('name').text = new_name
                                tree.write(file_path)
                                print(f"modify {file_path} name {old_name} to {new_name}")

        print(f"modify {self.xml_path}/* name {old_name} to {new_name} done")
        return True

    def modify(self,old_name,new_name):
        '''
        可以指定名称修改对应关系的情况
        '''
        for root_path, dirs, files in os.walk(self.xml_path):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root_path, file)
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    for obj in root.iter('object'):
                        name = obj.find('name').text
                        # if name == 'FieldWor':
                        # print(f"===> find {file_path} name {name} ")
                        if name == old_name:
                            # Check if the 'name_old' element exists
                            name_old = obj.find('name_old')
                            if name_old is None:
                                # If 'name_old' element does not exist, create it
                                name_old = ET.SubElement(obj, 'name_old')
                            # Set 'name_old' text to the old name
                            name_old.text = old_name

                            obj.find('name').text = new_name
                            tree.write(file_path)
                            print(f"modify {file_path} name {old_name} to {new_name}")
                            break
        print(f"modify {self.xml_path}/* name {old_name} to {new_name} done")
        return True

if __name__ == '__main__':

    ## For xml  dataset
    dataset_path = r'/home/lhf/DataSets/mnt_datasets/UAV_AG_all/365_AG/' #data_1_365
    data_labels  = r'/home/lhf/DataSets/mnt_datasets/UAV_AG_all/365_AG/ImageSet/labels'
    
    ### ========> Step 1. 分析数据集情况 获取类别分类(按照从数量多到少的顺序)
    label_summarizer = XMLLabelSummarizer(dataset_path)
    label_summarizer.print_all_labels()
    class_list = label_summarizer.get_class_list()
    print(class_list)

    ### ========> Step 2. xml 类别名修改
    # lebel_rename = XmlNameModify(dataset_path)
    # lebel_rename.modify('warcraft','plane')

    ### ========> Step 2. xml 转换为 txt
    # ['chariot', 'car', 'person', 'truck', 'tank', 'plane']
    new_class_list = ['J-vehicle','M-vehicle','person','plane']
    class_map_dist = {
        'chariot'   :'J-vehicle',
        'car'       :'M-vehicle',
        'person'    :'person',
        'truck'     :'M-vehicle',
        'tank'      :'J-vehicle',
        'plane'     :'plane'
    }
    label_convert = XMLtoTXTConverter(input_dir=dataset_path,out_dir=data_labels,class_list=new_class_list,class_map_dist=class_map_dist)
    # label_convert.convert()