import os
from tqdm import tqdm
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--data_root', default='data/occ3d')
    parser.add_argument('--output_root', default='data/occ3d/qwen_texts_step1')
    parser.add_argument('--start', type=int, default=251)
    parser.add_argument('--end', type=int, default=260)

    args = parser.parse_args()

    return args

def load_words(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            for word in line.strip().split(';'):
                if word.strip() and not word.startswith('<ref>'):
                    words.append(word.strip())
    words = list(set(words))
    return words

def process_words(word_list):
    processed_words = []
    for word in word_list:
        if not any(w in word and len(word) > len(w) and (word.endswith('s') or word.endswith('es')) for w in word_list if w!= word):
            processed_words.append(word)
    return processed_words

if __name__ == '__main__':
    args = parse_config()
    data_root = args.data_root
    gt_root = os.path.join(data_root, 'gts')
    output_root = args.output_root

    cams = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    
    print(f"start: {args.start}, {args.end}!")
    for i in tqdm(range(args.start, args.end)):
        scene = str(i).zfill(4)
        scene_name = 'scene-{}'.format(scene)
        print(f"Processing {scene_name}!")
        scene_dir = os.path.join(gt_root, scene_name)
        scene_words = []
        if os.path.exists(scene_dir):
            index_list = os.listdir(scene_dir)
            for index in tqdm(index_list):
                rec = nusc.get('sample', index)
                frame_words = []
                for cam_name in cams:
                    cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                    filename = cam_sample['filename'] # samples/CAM_BACK/image_name.jpg

                    img_path = os.path.join(data_root, filename)
                    # print("========start conversion=========")

                    text1 = """
                        This is an example image, where there exists these classes: 
                        traffic barrier; car; construction vehicle; crane;  pedestrian; traffic cone; road; sidewalk; terrain; grass; building; tree; fence; vegetaion; sky; traffic sign. 
                        Please carefully understand this image and these existing classes, and then describe the image in a brief paragraph.
                    """

                    query = tokenizer.from_list_format([
                        {'image': './example.jpg'},
                        {'text': text1},
                    ])
                    response, history = model.chat(tokenizer, query=query, history=None)

                    # print('response1: ', response)
                    
                    text2 = """
                        The image is captured by a camera on a driving car. 
                        Please carefully look at this image and detailedly describe the objects and background classes existed in this scene.
                    """
                    query2 = tokenizer.from_list_format([
                    {'image': img_path},
                    {'text': text2}
                    ])
                    # 2nd dialogue turn
                    response, history = model.chat(tokenizer, query=query2, history=history)

                    # print('response2: ', response)
                    text3 = """
                        Please list both the objects and background classes by a set of nouns.
                        Organize them in a fixed format, each noun is separated by ';'. 
                        The format is similar to the nouns listed from the initial example image: traffic barrier; car; construction vehicle; crane;  pedestrian; traffic cone; road; sidewalk; terrain; grass; building; tree; fence; vegetaion; sky; traffic sign
                    """
                    query3 = tokenizer.from_list_format([
                    {'image': img_path},
                    {'text': text3}
                    ])
                    
                    # 3rd dialogue turn
                    response, history = model.chat(tokenizer, query=query3, history=history)
                    # print('response3: ', response)
                    
                    os.makedirs(os.path.join(args.output_root, 'samples', cam_name), exist_ok=True)
                    with open(os.path.join(args.output_root, filename.replace('.jpg', '.txt')), 'w') as file:
                        file.write(response)