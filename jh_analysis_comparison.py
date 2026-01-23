from PIL import Image
import os
from tqdm import tqdm
import json

def get_images_from_folder(folder):
    images = []
    file_names = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_names.append(filename)
    
    # Sort filenames based on the integer value after "frame_"
    file_names.sort(key=lambda x: int(x.split('frame_')[-1].split('.')[0]))
    
    for filename in file_names:
        img_path = os.path.join(folder, filename)
        images.append(Image.open(img_path))
    
    return images, file_names

def combine_images(img1, img2):
    width1, height1 = img1.size
    width2, height2 = img2.size

    combined_width = width1 + width2
    combined_height = max(height1, height2)

    combined_img = Image.new('RGB', (combined_width, combined_height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (width1, 0))

    return combined_img

def get_folders_from_path(path):
    folders = [f.path for f in os.scandir(path) if f.is_dir()]
    folders.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    return folders

def extract_task(path):
    with open(path, 'r') as f:
        task = json.load(f)
    video_titles = task['videos'].keys()
    extracted_texts = [title.split('_', 3)[-1] for title in video_titles]
    return extracted_texts
    

if __name__ == "__main__":
    folder1 = '/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor/JPEGImages_Sparse/val'
    folder2 = '/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor/Annotations_Sparse/val'
    folder3 = '/home/jihun/workspace/repositories/vos_task/actionvos/ReferFormer/actionvos_dirs/r101/val_org'
    folder4 = '/home/jihun/workspace/repositories/vos_task/actionvos/ReferFormer/actionvos_dirs/r101/val_seim_ver1'
    task_description = '/home/jihun/workspace/repositories/vos_task/actionvos/dataset_visor/ImageSets/val_meta_expressions_promptaction.json'
    output_folder = '/home/jihun/workspace/repositories/vos_task/actionvos/jh_comparison_collections/jh-comparison-seim_ver1'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfolders1 = get_folders_from_path(folder1)
    subfolders2 = get_folders_from_path(folder2)
    subfolders3 = get_folders_from_path(folder3)
    subfolders4 = get_folders_from_path(folder4)

    size1 = len(subfolders1)
    size2 = len(subfolders2)
    size3 = len(subfolders3)
    size4 = len(subfolders4)
    assert size1 == size2 == size3 == size4

    for j in tqdm(range(size1), desc="Processing images"):
        # j = 14
        task = extract_task(task_description)
        # import ipdb; ipdb.set_trace()
        images1, file_names1 = get_images_from_folder(subfolders1[j])
        images2, file_names2 = get_images_from_folder(subfolders2[j])
        # breakpoint()
        images3, file_names3 = get_images_from_folder(subfolders3[j])
        images4, file_names4 = get_images_from_folder(subfolders4[j])

        num_files = len(file_names1)

        for i in range(num_files):
            combined_img1 = combine_images(Image.open(os.path.join(subfolders1[j], file_names1[i])), Image.open(os.path.join(subfolders2[j], file_names2[i])))
            combined_img2 = combine_images(combined_img1, Image.open(os.path.join(subfolders3[j], file_names3[i])))
            combined_img3 = combine_images(combined_img2, Image.open(os.path.join(subfolders4[j], file_names4[i])))
            combined_img3.save(os.path.join(output_folder, f'{j+1}_comparison_{i+1}_{task[j]}.png'))
        # for i, (img1, img2, img3) in enumerate(zip(images1, images2, images3)):
        #     if file_names1[i] == file_names2[i] == file_names3[i]:
        #         combined_img1 = combine_images(img1[i], img2[i])
        #         combined_img = combine_images(combined_img1, img3[i])
        #         combined_img.save(os.path.join(output_folder, f'{j+1}_comparison_{i+1}_{task[j]}.png'))
        
        # breakpoint()
        # breakpoint()