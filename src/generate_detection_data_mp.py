# library for rdkit application
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdCoordGen, rdAbbreviations, AllChem
from rdkit.Chem.Draw import MolDrawOptions

# others
import pandas as pd
import matplotlib.pyplot as plt
import cv2, random, copy, os, math, glob2
import numpy as np
import warnings
from PIL import Image
from typing import List, Optional
from tqdm.auto import tqdm
import fitz
from io import BytesIO

# Multi-processing method
import multiprocessing as mp
from functools import partial

from src.generate_detection_data import Smiles2Img, generate_integrated_image, Table2Img, prepare_background_imgs, generate_PDF_image

warnings.filterwarnings(action = "ignore")

def process(
    proc : int,
    n_procs:int,
    smiles_list : List, 
    table_list : List, 
    num_data : int = 200000, 
    max_smiles : int = 12, 
    max_table : int = 2, 
    img_size : int = 128, 
    fig_height : int = 900, 
    fig_width : int = 600, 
    background_pdf_list:Optional[List] = None
    ):
    
    col_ids = []
    col_smiles = []
    col_table = []
    col_img = []
    col_n_molecules = []
    col_classes = []
    col_label = []
    
    if not os.path.exists("./dataset/detection/folder_{}".format(str(proc).zfill(2))):
        os.mkdir("./dataset/detection/folder_{}".format(str(proc).zfill(2)))
    
    print("Data generation process : {} proc".format(proc+1))
    
    num_data_proc = num_data // n_procs
    
    indx = num_data_proc * proc
    pdf_indx = len(background_pdf_list) * proc // n_procs
    
    if proc < n_procs - 1:
        smiles_list = smiles_list[proc * len(smiles_list) // n_procs : (proc + 1) * len(smiles_list) // n_procs]
        ni = num_data_proc * proc
        nf = num_data_proc * (proc + 1)
            
    else:
        smiles_list = smiles_list[proc * len(smiles_list) // n_procs : ]
        ni = num_data_proc * proc
        nf = num_data
    
    for num_img in tqdm(range(ni, ni + num_data_proc // 3), 'Data generation process (1)', disable=disable_tqdm):
        
        n_smiles = random.randint(1, max_smiles+1)
        random.shuffle(smiles_list)
        sampled_smiles_list = smiles_list[0:n_smiles]
        smiles_images = [Smiles2Img(smiles, img_size) for smiles in sampled_smiles_list]
        
        integrated_image, label = generate_integrated_image(smiles_images, fig_height, fig_width)
        classes = [3 for _ in range(len(smiles_images))]

        if integrated_image is not None:
            indx += 1
            
            img_dir = os.path.join("./dataset/detection/folder_{}".format(str(proc).zfill(2)), "img_{}.jpg".format(str(indx).zfill(5)))
            cv2.imwrite(img_dir, integrated_image)
            
            col_ids.append(num_img)
            col_smiles.append(sampled_smiles_list)
            col_table.append([])
            col_img.append(img_dir)
            col_n_molecules.append(n_smiles)
            col_label.append(label)
            col_classes.append(classes)
    
    for num_img in tqdm(range(ni + num_data_proc // 3, ni + num_data_proc * 2 // 3), 'Data generation process (2)', disable=disable_tqdm):
        
        n_table = random.randint(1, max_table+1)
        n_smiles = random.randint(1, max_smiles+1)

        random.shuffle(table_list)
        sampled_table_list = table_list[0:n_table]
        
        n_smiles = random.randint(1, max_smiles+1)
        random.shuffle(smiles_list)
        sampled_smiles_list = smiles_list[0:n_smiles]
        smiles_images = [Smiles2Img(smiles, img_size) for smiles in sampled_smiles_list]
        
        max_h = fig_height*0.35
        max_w = fig_width*0.35
        min_h = fig_height*0.1
        min_w = fig_width*0.1

        table_images = [Table2Img(path, max_h, max_w, min_h, min_w) for path in sampled_table_list]
        
        background_images, background_classes = prepare_background_imgs("./dataset/sample_text", "./dataset/sample_img", fig_height, fig_width, 2, 4)
            
        integrated_image, label = generate_PDF_image(table_images + smiles_images, background_images, fig_height, fig_width, None)
        classes = background_classes + [4 for _ in range(len(table_images))] + [3 for _ in range(len(smiles_images))]
     
        if integrated_image is not None:
            indx += 1
            
            img_dir = os.path.join("./dataset/detection/folder_{}".format(str(proc).zfill(2)), "img_{}.jpg".format(str(indx).zfill(5)))
            cv2.imwrite(img_dir, integrated_image)
            
            col_ids.append(num_img)
            col_smiles.append(sampled_smiles_list)
            col_table.append(sampled_table_list)
            col_img.append(img_dir)
            col_n_molecules.append(n_smiles)
            col_label.append(label)
            col_classes.append(classes)
            
    for num_img in tqdm(range(ni + num_data_proc * 2 // 3, nf), 'Data generation process (3)', disable=disable_tqdm):
        
        n_table = random.randint(1, max_table+1)
        n_smiles = random.randint(1, max_smiles+1)

        random.shuffle(table_list)
        sampled_table_list = table_list[0:n_table]
        
        n_smiles = random.randint(1, max_smiles+1)
        random.shuffle(smiles_list)
        sampled_smiles_list = smiles_list[0:n_smiles]
        smiles_images = [Smiles2Img(smiles, img_size) for smiles in sampled_smiles_list]
        
        max_h = fig_height*0.35
        max_w = fig_width*0.35
        min_h = fig_height*0.1
        min_w = fig_width*0.1

        table_images = [Table2Img(path, max_h, max_w, min_h, min_w) for path in sampled_table_list]
        
        background_images, background_classes = prepare_background_imgs("./dataset/sample_text", "./dataset/sample_img", fig_height, fig_width, 2, 4)
            
        pdf_background = background_pdf_list[pdf_indx]                
        integrated_image, label = generate_PDF_image(table_images + smiles_images, background_images, fig_height, fig_width, pdf_background)
        classes = background_classes + [4 for _ in range(len(table_images))] + [3 for _ in range(len(smiles_images))]
     
        if integrated_image is not None:
            indx += 1
            pdf_indx += 1
            
            if pdf_indx >= len(background_pdf_list) * (proc + 1) // n_procs - 1:
                pdf_indx = len(background_pdf_list) * proc // n_procs
            
            img_dir = os.path.join("./dataset/detection/folder_{}".format(str(proc).zfill(2)), "img_{}.jpg".format(str(indx).zfill(5)))
            cv2.imwrite(img_dir, integrated_image)
            
            col_ids.append(num_img)
            col_smiles.append(sampled_smiles_list)
            col_table.append(sampled_table_list)
            col_img.append(img_dir)
            col_n_molecules.append(n_smiles)
            col_label.append(label)
            col_classes.append(classes)
            
    df = pd.DataFrame({
        "id":col_ids,
        "SMILES":col_smiles,
        "table":col_table,
        "img":col_img,
        "n_molecules":col_n_molecules,
        "label":col_label,
        "class":col_classes
    })
    
    df.to_csv("./dataset/multiprocessing/data_label_{}.csv".format(proc), index = False)
    print("Data generation process complete : {} proc".format(proc+1))
    return

if __name__=="__main__":
    
    n_procs = mp.cpu_count()
    
    print("Multi-processing n_procs : {}".format(n_procs))

    # Setup
    disable_tqdm = True
    num_data = n_procs * 3 * 2048
    save_dir = "./dataset/detection"
    
    print("Total dataset : {}".format(num_data))
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    if not os.path.exists("./dataset/multiprocessing"):
        os.mkdir("./dataset/multiprocessing")
    
    df = pd.read_csv('./dataset/surechembl_cleansed.csv').sample(n=800000)
    
    smiles_list = df['SMILES'].to_list()
    table_list = random.sample(glob2.glob('./dataset/cropped_table_images/*'), 256)
    
    # Background pdf loaded
    dir_list_pdf = glob2.glob("./dataset/sample_background/patents_after_20230601/*")
    print("PDF background list: ", len(dir_list_pdf))

    dir_list_pdf = random.sample(dir_list_pdf, 128)
    background_pdf_list = []
    
    print("PDF image extraction...")
    for filepath in tqdm(dir_list_pdf, "PDF image extraction", disable=disable_tqdm):
        
        doc = fitz.open(filepath)
        imgs = []

        for i, page in enumerate(doc):
            img = page.get_pixmap().tobytes()
            img = BytesIO(img)
            img = Image.open(img)
            img = img.convert('RGB').resize((600, 900))
            imgs.append(np.array(img))
        
        background_pdf_list.extend(imgs)
        
    print("PDF image extraction complete")
    
    # Multi-processing for generating detection data
    pool = mp.Pool(processes=n_procs)
    
    process_per_proc = partial(
        process,
        n_procs = n_procs,
        smiles_list = smiles_list,
        table_list = table_list,
        num_data = num_data,
        max_smiles = 12,
        max_table = 1,
        img_size = 112,
        fig_height = 900,
        fig_width = 600,
        background_pdf_list = background_pdf_list
    )
    
    pool.map(
        process_per_proc, 
        [proc for proc in range(n_procs)]
    )
    
    pool.close()
    pool.join()
    
    # Merge files 
    df_detection = None
    for proc in range(n_procs):
        df = pd.read_csv("./dataset/multiprocessing/data_label_{}.csv".format(int(proc)))

        if proc == 0:
            df_detection = df
        else:
            df_detection = pd.concat([df_detection, df], axis = 0)
    
    df_detection.to_csv("./dataset/detection_data.csv", index = False)
    
    print("Data generation process clear")