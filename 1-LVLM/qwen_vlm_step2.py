import os
from tqdm import tqdm
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--data_root', default='data/occ3d')
    parser.add_argument('--source_root', default='qwen_texts_step1')
    parser.add_argument('--output_root', default='data/occ3d/qwen_texts_step2')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=1111)

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

if __name__ == '__main__':
    args = parse_config()
    data_root = args.data_root
    gt_root = os.path.join(data_root, 'gts')
    output_root = args.output_root

    cams = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    
    print(f"start: {args.start}, {args.end}!")
    for i in tqdm(range(args.start, args.end)):
        scene = str(i).zfill(4)
        scene_name = 'scene-{}'.format(scene)
        print(f"Processing {scene_name}!")
        scene_dir = os.path.join(gt_root, scene_name)
        if os.path.exists(scene_dir):
            index_list = os.listdir(scene_dir)
            for index in tqdm(index_list):
                rec = nusc.get('sample', index)
                frame_words = []
                for cam_name in cams:
                    cam_sample = nusc.get('sample_data', rec['data'][cam_name])
                    filename = cam_sample['filename']

                    img_path = os.path.join(data_root, filename)
                    text_path = img_path.replace('samples', args.source_root).replace('.jpg', '.txt')
                    words = load_words(text_path)
                    frame_words += words
                frame_words = list(set(frame_words))

                kept_words = []
                
                for kept_word in frame_words:
                    if kept_word.strip() and len(kept_word.strip().split()) <= 2 :
                        kept_words.append(kept_word.strip().strip('.').strip('\n').lower())

                kept_words = list(set(kept_words))

                os.makedirs(os.path.join(output_root, scene_name), exist_ok=True)
                with open(os.path.join(output_root, scene_name, index + '.txt'), 'w') as f:
                    for item in kept_words:
                        f.write(item + '\n')