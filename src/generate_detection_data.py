'''
    Preprocessing code for generating the dataset for training detection model
    This code is based on preprocessing code developed by Hanbit Kim
    
    Basically, we use functions for generating augmented molecular image from SMILES dataset
    However, we additionally implemented image with multi-objects (molecular image) in common figure
    
    The dataset consists of 
    - Collection molecular images for detection model saved in ./dataset/images/detection
    - Directory for saved images
    - Positional information of the molecules in each image (bbox, categories) : [(Xm, Ym, H, W),(),...]
'''

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

warnings.filterwarnings(action = "ignore")

# compute internal energy of molecular configuration
def internal_energy_mod(mol):
    mol.GetConformer()
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    ffm = AllChem.MMFFGetMoleculeForceField(mol, mp)
    return ffm.CalcEnergy()

# rotation using bond information
def rotable_bond_rotation(mol:Chem.rdchem.Mol,bond, angle):
    start,end = bond
    new_mol = copy.deepcopy(mol)
    for atom in new_mol.GetAtomWithIdx(start).GetNeighbors():
        if atom.GetIdx()!=end:
            i = atom.GetIdx()
            break
    for atom in new_mol.GetAtomWithIdx(end).GetNeighbors():
        if atom.GetIdx()!=start:
            l = atom.GetIdx()
            break
        
    AllChem.EmbedMolecule(new_mol)
    AllChem.Compute2DCoords(new_mol)
    conf = new_mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf,i,start,end,l,angle)
    return new_mol

# Flip molecular image
def get_flipped_mol(mol:Chem.rdchem.Mol):
    rot_y = np.array([[-1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,1.]]) # rotate 180 about y axis
    rdMolTransforms.TransformConformer(mol.GetConformer(0), rot_y)
    return mol

# Rotate molecular image
def get_rotated_mol(mol:Chem.rdchem.Mol):
    RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rotatable_bonds = mol.GetSubstructMatches(RotatableBond)
    if len(rotatable_bonds)==0:
        return mol
    count=0 
    while True:
        try:
            random_rotatebond = random.choice(rotatable_bonds)
            new_mol = rotable_bond_rotation(mol,random_rotatebond, math.pi)
            energy_mol = copy.deepcopy(new_mol)
            internal_energy = internal_energy_mod(energy_mol)
        except AttributeError as e:
            # print(e)
            return mol
        except Exception as e:
            # print(e)
            break
        except rdkit.Chem.rdchem.KekulizeException as e:
            # print(e)
            return mol
        except RuntimeError as e:
            # print(e)
            return mol
        if internal_energy<10000:
            mol = new_mol
            break
        count+=1
        if count>10:
            return mol
    return mol

# abbreviation of molecular image
def get_abbreviationed_mol(mol:Chem.rdchem.Mol):
    mol_img = copy.deepcopy(mol)
    abbrevs = rdAbbreviations.GetDefaultAbbreviations()
    mol_img = rdAbbreviations.CondenseMolAbbreviations(mol_img,abbrevs)
    return mol_img

# insert padding to the molecular image
def get_padded_img(img_np:np.ndarray):
    top = random.randint(0, 20)
    bottom = random.randint(0, 20)
    right = random.randint(0, 20)
    left = random.randint(0, 20)
    image_padded = cv2.copyMakeBorder(src = img_np, 
                                      top = top, 
                                      bottom = bottom, 
                                      left = left,
                                      right = right,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value = [255, 255, 255]
                                      )
    return image_padded

# get rotated molecular image with 90 degree
def get_rotated_img(img:np.ndarray):
    if random.randint(0,1):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# add noise to the molecular image
def get_gaussian_noised_img(img:np.ndarray, IMG_SIZE : int = 128):
    noise = np.random.normal(0, random.randint(0,10), size=(IMG_SIZE, IMG_SIZE))
    return img+noise

# save image as jpeg + remove augmentation?
def get_jpg_augmented_img(img_path):
    im = Image.open(img_path)
    im.save(f'{img_path[:-4]}.jpg')
    im = Image.open(f'{img_path[:-4]}.jpg')
    im.save(img_path)
    os.remove(f'{img_path[:-4]}.jpg')
    
def getDrawingOptions():
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.rotate = int(round(random.uniform(0, 360),0))
    options.bondLineWidth = random.randint(1,4)
    options.scaleBondWidth = True
    options.minFontSize = int(round(random.uniform(6, 17),0))
    options.multipleBondOffset = round(random.uniform(0.08, 0.16),2)
    options.additionalAtomLabelPadding = round(random.uniform(0, 0.13),2)
    options.padding = round(random.uniform(0.04, 0.15), 2)
    options.explicitMethyl = not bool(random.randint(0,3))
    options.comicMode = not bool(random.randint(0,1))
    font_path = 'font_files'
    font_list = os.listdir(font_path)
    options.fontFile = font_path + '/' + font_list[random.randint(0,len(font_list)-1)]
    options.addStereoAnnotation = not bool(random.randint(0,3))
    options.annotationFontScale = round(random.uniform(1.0, 0.6),2)
    options.atomLabelDeuteriumTritium = True

    return options

# convert smiles text to image
def Smiles2Img(smiles, IMG_SIZE : int = 128):
    
    params = Chem.SmilesParserParams()
    params.removeHs = False
    
    mol = Chem.MolFromSmiles(smiles, params)
    rdCoordGen.AddCoords(mol)
    
    ########## flip #########
    flip_molecule = random.randint(0,3)
    if not flip_molecule:
        mol = get_flipped_mol(mol)
    
    ########## rotation ##########
    rotate_molecule = random.randint(0,3)
    if rotate_molecule:
        mol = Chem.AddHs(mol)
        mol = get_rotated_mol(mol) # addHs occur
    
    use_abbreviation = random.randint(0,3)
    if not use_abbreviation:
        mol = get_abbreviationed_mol(mol)

    options = getDrawingOptions()
    img = Chem.Draw.MolToImage(mol, size=(IMG_SIZE+random.randint(-32, 64), IMG_SIZE+random.randint(-32, 64)), options=options)
    img = np.array(img)
    
    return img

def Table2Img(path, max_h, max_w, min_h, min_w):
    img = cv2.imread(path)
    h, w, _ = img.shape

    max_scale_factor = min(max_h/h, max_w/w)
    min_scale_factor = max(min_h/h, min_w/w)

    if max_scale_factor < min_scale_factor:
        scale_factor = max_scale_factor
    else:            
        scale_factor = random.uniform(min_scale_factor, max_scale_factor)

    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return img

def center2corner(xc,yc,W,H):
    x1 = xc - 0.5 * W
    y1 = yc - 0.5 * H
    x2 = xc + 0.5 * W
    y2 = yc + 0.5 * H
    return [int(x1), int(y1), int(x2), int(y2)]

def corner2center(x1,y1,x2,y2):
    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    W = abs(x1-x2)
    H = abs(y1-y2)
    return [int(xc), int(yc), int(W), int(H)]

def split_region(integrated_image:np.ndarray, n_object:int):
    
    H = integrated_image.shape[0]
    W = integrated_image.shape[1]
    
    obj2reg = {
        1 : 0,
        2 : 1,
        3 : 1,
        4 : 2,
        5 : 2,
        6 : 3, 
        7 : 3,
        8 : 4,
        9 : 4,
        10 : 5,
        11 : 5,
        12 : 5,
    }

    if n_object <= 12:
        region_type = obj2reg[n_object]
    else:
        region_type = 5
    
    # pbox = [x1, y1, x2, y2]
    if region_type == 0:
        pboxs = [[0,0,W,H]]
    elif region_type == 1:
        pboxs = [[0,0,int(W/2),H], [int(W/2),0,W,H]]
    elif region_type == 2:
        pboxs = [[0,0,int(W/2),int(H/2)],[0,int(H/2),int(W/2),H],[int(W/2),int(H/2),W,H],[int(W/2),0,W,int(H/2)]]
    elif region_type == 3:
        pboxs = [
            [0,0,int(W/2),int(H/3)],
            [0,int(H/3),int(W/2),int(2*H/3)],
            [0,int(2*H/3),int(W/2),H],
            [int(W/2),int(2*H/3),W,H],
            [int(W/2),int(H/3),W,int(2*H/3)],
            [int(W/2),0,W,int(H/3)],
        ]
        
    elif region_type == 4:
        pboxs = [
            [0,0,int(W/2),int(H/4)],
            [0,int(H/4),int(W/2),int(2*H/4)],
            [0,int(2*H/4),int(W/2),int(3*H/4)],
            [0,int(3*H/4),int(W/2),H],
            [int(W/2),int(3*H/4),W,H],
            [int(W/2),int(2*H/4),W,int(3*H/4)],
            [int(W/2),int(H/4),W,int(2*H/4)],
            [int(W/2),0,W,int(H/4)],
        ]
    
    elif region_type == 5:
        pboxs = [
            [0,0,int(W/3),int(H/4)],
            [0,int(H/4),int(W/3),int(2*H/4)],
            [0,int(2*H/4),int(W/3),int(3*H/4)],
            [0,int(3*H/4),int(W/3),H],
            [int(W/3),0,int(2*W/3),int(H/4)],
            [int(W/3),int(H/4),int(2*W/3),int(2*H/4)],
            [int(W/3),int(2*H/4),int(2*W/3),int(3*H/4)],
            [int(W/3),int(3*H/4),int(2*W/3),H],
            [int(2*W/3),int(3*H/4),W,H],
            [int(2*W/3),int(2*H/4),W,int(3*H/4)],
            [int(2*W/3),int(H/4),W,int(2*H/4)],
            [int(2*W/3),0,W,int(H/4)],
        ]  

    return pboxs

# generate the canvas for training detection model
# Molecular images only
def generate_integrated_image(images, fig_height : int = 900, fig_width : int = 600, integrated_image:Optional[np.ndarray] = None):
    
    # Create an empty canvas for the integrated image
    if integrated_image is None:
        integrated_image = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255
        
    n_mols = len(images)
    region_boxes = split_region(integrated_image, n_mols)
    region_boxes_ordered = np.random.permutation(region_boxes)
    
    images_list = []
    label = []
    
    for idx, image in enumerate(images):
        
        if idx >= len(region_boxes):
            region_box = region_boxes[idx-len(region_boxes)]
        else:
            region_box = region_boxes[idx]
        
        xl,yl,xr,yr = region_box[0], region_box[1], region_box[2], region_box[3]
        
        W = xr - xl
        H = yr - yl
        
        w = image.shape[1]
        h = image.shape[0]
        
        if w >= W:
            image = image[:,int(w/2)-int(W/2):int(w/2)+int(W/2)]
            w = W
        
        if h >= H:
            image = image[int(h/2) - int(H/2):int(h/2)+int(H/2),:]
            h = H
        
        x = np.random.randint(xl, xr - w + 1)
        y = np.random.randint(yl, yr - h + 1)
        
        label.append([x,y,x+w,y+h])
        
        # Resize the image to its height and width specified in COCO format
        image = cv2.resize(image, (w, h))

        integrated_image[y:y+h, x:x+w] = image
        
    return integrated_image, label
        
# generate the canvas for training detection model with image and text : virtual PDF file image
def generate_PDF_image(images, background_images, fig_height : int = 900, fig_width : int = 600, pdf_background=None):
    
    # Create an empty canvas for the integrated image
    if pdf_background is not None:
        integrated_image = pdf_background
    else:
        integrated_image = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 255
    
    # Process 1 : add image and text data
    integrated_image, background_img_label = generate_integrated_image(background_images, fig_height, fig_width, integrated_image)
     
    # Process 2 : add molecular image to the canvas
    # In this case, molecular image can overlap the canvas whether the background image exist or does not on it.
    integrated_image, molecule_label = generate_integrated_image(images, fig_height, fig_width, integrated_image)
    
    label = []
    label.extend(background_img_label)
    label.extend(molecule_label)
        
    return integrated_image, label

def prepare_background_imgs(text_folder : str, img_folder : str, fig_height : int, fig_width : int, n_min : int = 2, n_max : int = 4):
    background_imgs = []
    background_classes = [1 for _ in range(len(glob2.glob(os.path.join(text_folder, "*"))))] + [2 for _ in range(len(glob2.glob(os.path.join(img_folder, "*"))))]
    
    n_object = np.random.randint(n_min, n_max + 1)
    path_list = glob2.glob(os.path.join(text_folder, "*")) + glob2.glob(os.path.join(img_folder, "*"))
    
    indices = random.sample([i for i in range(len(path_list))], n_object)
    
    path_list = [path_list[idx] for idx in indices]
    background_classes = [background_classes[idx] for idx in indices]
    
    max_h = int(fig_height * 0.5)
    max_w = int(fig_width * 0.5)
    
    for path in path_list:
        img = Image.open(path, mode = 'r')
        img = img.convert('RGB')
        
        img = cv2.imread(path)
        h,w, _ = img.shape
        
        if w >= h and w > max_w:
            w_new = max_w
            h_new = int(h * max_w / w)
            img = cv2.resize(img, (w_new, h_new))
            
        elif h >= w and h > max_h:
            h_new = max_h
            w_new = int(w * max_h / h)
            img = cv2.resize(img, (w_new, h_new))
            
        elif h >= w and w > max_w:
            w_new = max_w * 0.5
            h_new = int(h * max_w / w)
            img = cv2.resize(img, (w_new, h_new))
            
        background_imgs.append(img)
    
    return background_imgs, background_classes

def process(
    smiles_list : List, 
    table_list : List, 
    num_data : int = 200000, 
    max_smiles : int = 12, 
    max_table : int = 2, 
    img_size : int = 128, 
    fig_height : int = 900, 
    fig_width : int = 600, 
    save_dir : str = "./dataset/detection", 
    background_pdf_list:Optional[List] = None
    ):
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    col_ids = []
    col_smiles = []
    col_table = []
    col_img = []
    col_n_molecules = []
    col_classes = []
    col_label = []
    
    indx = 0
    pdf_indx = 0
    
    # Generation process
    # (1) Molecular images only with zero background
    # (2) Molecular images and other objects with zero background
    # (3) Molecular images and other objects with PDF background
    
    for num_img in tqdm(range(num_data//3), 'Data generation process (1)', disable=disable_tqdm):
        
        n_smiles = random.randint(1, max_smiles+1)
        random.shuffle(smiles_list)
        sampled_smiles_list = smiles_list[0:n_smiles]
        smiles_images = [Smiles2Img(smiles, img_size) for smiles in sampled_smiles_list]
        
        integrated_image, label = generate_integrated_image(smiles_images, fig_height, fig_width)
        classes = [3 for _ in range(len(smiles_images))]

        if integrated_image is not None:
            indx += 1
            
            img_dir = os.path.join(save_dir, "img_{}.jpg".format(str(indx).zfill(5)))
            cv2.imwrite(img_dir, integrated_image)
            
            col_ids.append(num_img)
            col_smiles.append(sampled_smiles_list)
            col_table.append([])
            col_img.append(img_dir)
            col_n_molecules.append(n_smiles)
            col_label.append(label)
            col_classes.append(classes)
    
    for num_img in tqdm(range(num_data//3, 2 * num_data//3), 'Data generation process (2)', disable=disable_tqdm):
        
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
            
            img_dir = os.path.join(save_dir, "img_{}.jpg".format(str(indx).zfill(5)))
            cv2.imwrite(img_dir, integrated_image)
            
            col_ids.append(num_img)
            col_smiles.append(sampled_smiles_list)
            col_table.append(sampled_table_list)
            col_img.append(img_dir)
            col_n_molecules.append(n_smiles)
            col_label.append(label)
            col_classes.append(classes)
            
    for num_img in tqdm(range(2*num_data//3, num_data), 'Data generation process (3)', disable=disable_tqdm):
        
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
            
            if pdf_indx >= len(background_pdf_list) - 1:
                pdf_indx = 0
            
            img_dir = os.path.join(save_dir, "img_{}.jpg".format(str(indx).zfill(5)))
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
    
    return df

# generate detection dataset
if __name__=="__main__":
    
    disable_tqdm = True
    
    df = pd.read_csv('./dataset/surechembl_cleansed.csv').sample(n=100000)
    smiles_list = df['SMILES'].to_list()

    table_list = random.sample(glob2.glob('./dataset/cropped_table_images/*'), 256)
    
    # Background pdf loaded
    dir_list_pdf = glob2.glob("./dataset/sample_background/patents_after_20230601/*")
    print("PDF background list: ", len(dir_list_pdf))

    dir_list_pdf = random.sample(dir_list_pdf, 128)
    background_pdf_list = []

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
    
    df_detection = process(smiles_list, table_list, num_data=12000, max_smiles=12, max_table=1, img_size=112, fig_height = 900, fig_width=600, save_dir = "./dataset/detection", background_pdf_list = background_pdf_list)
    df_detection.to_csv("./dataset/detection_data.csv")
    
    print("Data generation process clear")