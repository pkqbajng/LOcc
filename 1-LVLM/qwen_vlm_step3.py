import re
import os
import webcolors
from copy import deepcopy
from tqdm import tqdm
from argparse import ArgumentParser
from nuscenes.nuscenes import NuScenes

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--data_root', default='data/occ3d')
    parser.add_argument('--source_root', default='qwen_texts_step2')
    parser.add_argument('--output_root', default='./qwen_texts')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=1111)

    args = parser.parse_args()

    return args

def load_txt(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            for word in line.strip().split(';'):
                if word.strip() and not word.startswith('<ref>'):
                    words.append(word.strip())
    words = list(set(words))
    return words

def is_color(word):
    try:
        webcolors.name_to_rgb(word)
        return True
    except ValueError:
        return False

def remove_color(word):
    parts = word.split()
    new_parts = []
    for part in parts:
        if not is_color(part):
            new_parts.append(part)
    
    if len(new_parts) > 0:
        result = " ".join(new_parts)
    else:
        result = None
    return result

def has_number(word):
    pattern = r'\d'
    return bool(re.search(pattern, word))

def has_special_char(word):
    for char in word:
        if not char.isalpha() and not char.isdigit() and char!= " ":
            return True
    return False

def has_special_token(word):
    return has_number(word) and has_special_char(word)

def post_processing(texts):
    words = []
    for text in texts:
        word = text.strip('*-').strip()
        if word.isdigit():
            continue
        if word == '':
            continue
        if "'s" in word:
            continue
        if "no " in word:
            continue
        if "'" in word:
            continue
        if "sky" in word:
            continue
        if "cloud" in word:
            continue
        
        word = remove_color(word)
        if word is not None and not has_special_token(word):
            words.append(word.strip())
    
    return list(set(words))

def plural_to_singular(word, reference_words):
    if word.endswith('s'):
        for other_word in reference_words:
            if word[:-1] == other_word:
                word = other_word
                break
        parts = word.split()
        if len(parts) == 1:
            return word
        else:
            for other_word in reference_words:
                if parts[-1][:-1] == other_word:
                    parts[-1] = other_word
                    break
            return ' '.join(parts).strip()
    elif word.endswith('es'):
        for other_word in reference_words:
            if word[:-2] == other_word:
                word = other_word
                break
        parts = word.split()
        if len(parts) == 1:
            return word
        else:
            for other_word in reference_words:
                if parts[-1][:-2] == other_word:
                    parts[-1] = other_word
                    break
            return ' '.join(parts).strip()
    else:
        return word

def filter_words(texts, reference_texts):
    words = []
    for text in texts:
        text = plural_to_singular(text, reference_texts)
        words.append(text)
    
    return list(set(words))

        
if __name__ == '__main__':
    args = parse_config()
    data_root = args.data_root
    gt_root = os.path.join(data_root, 'gts')
    output_root = args.output_root

    device = "cuda"
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=data_root,
                    verbose=True)
    min_length = 100
    max_length = 10
    print(f"start: {args.start}, {args.end}!")
    for i in tqdm(range(args.start, args.end)):
        scene = str(i).zfill(4)
        scene_name = 'scene-{}'.format(scene)
        # print(f"Processing {scene_name}!")
        scene_dir = os.path.join(gt_root, scene_name)
        scene_words = []
        if os.path.exists(scene_dir):
            index_list = os.listdir(scene_dir)
            frame_words_list = dict()

            # for index in tqdm(index_list):
            for index in index_list:
                rec = nusc.get('sample', index)
                text_path = os.path.join(args.source_root, scene_name, index + '.txt')
                frame_words = load_txt(text_path)

                pre_rec = nusc.get('sample', index)
                next_rec = nusc.get('sample', index)
                while len(frame_words) <= 3:
                    pre_index = pre_rec['prev']
                    if pre_index == '':
                        break
                    pre_rec = nusc.get('sample', pre_index)
                    text_path = os.path.join(args.source_root, scene_name, pre_index + '.txt')
                    frame_words = load_txt(text_path)
                while len(frame_words)  <= 3:
                    next_index = next_rec['next']
                    if next_index == '':
                        break
                    next_rec = nusc.get('sample', next_index)
                    text_path = os.path.join(args.source_root, scene_name, next_index + '.txt')
                    frame_words = load_txt(text_path)
                frame_words = list(set(frame_words))
                frame_words = post_processing(frame_words)
                frame_words.sort()

                frame_words_list[index] = frame_words
                scene_words += frame_words
            
            scene_words = list(set(scene_words))
            scene_words = post_processing(scene_words)
            reference_words = deepcopy(scene_words)
            scene_words = filter_words(scene_words, reference_words)

            new_scene_words = []
            new_frame_words_list = dict()
            for index in index_list:
                rec = nusc.get('sample', index)
                frame_words = frame_words_list[index]
                frame_words = filter_words(frame_words, reference_words)
                
                prev = rec['prev']
                if prev != '':
                    prev_words = frame_words_list[prev]
                    prev_words = filter_words(prev_words, scene_words)
                    frame_words = list(set(frame_words) & set(prev_words))
                else:
                    next = rec['next']
                    next_words = frame_words_list[next]
                    next_words = filter_words(next_words, scene_words)
                    frame_words = list(set(frame_words) & set(next_words))
                new_frame_words_list[index] = frame_words
            
            for index in index_list:
                rec = nusc.get('sample', index)
                frame_words = new_frame_words_list[index]

                pre_rec = nusc.get('sample', index)
                next_rec = nusc.get('sample', index)

                while len(frame_words) <= 10:
                    pre_index = pre_rec['prev']
                    if pre_index == '':
                        break
                    pre_rec = nusc.get('sample', pre_index)
                    prev_words = new_frame_words_list[pre_index]
                    frame_words += prev_words
                    frame_words = list(set(frame_words))

                while len(frame_words)  <= 10:
                    next_index = next_rec['next']
                    if next_index == '':
                        break
                    next_rec = nusc.get('sample', next_index)
                    next_words = new_frame_words_list[next_index]
                    frame_words += next_words
                    frame_words = list(set(frame_words))

                if len(frame_words) < min_length:
                    min_length = len(frame_words)
                    print('min_length: ', min_length)
                
                frame_words.sort()
                new_scene_words += frame_words
                os.makedirs(os.path.join(output_root, scene_name), exist_ok=True)
                with open(os.path.join(output_root, scene_name, index + '.txt'), 'w') as f:
                    for item in frame_words:
                        f.write(item + '\n')
            
            new_scene_words = list(set(new_scene_words))
            new_scene_words.sort()

            if len(new_scene_words) > max_length:
                max_length = len(new_scene_words)
                print('max_length:', max_length)
            
            with open(os.path.join(output_root, scene_name, 'scene_vocabulary.txt'), 'w') as f:
                for item in new_scene_words:
                    f.write(item + '\n')