import os
import shutil

# Percorso della cartella principale "Training"
#base_dir = 'Dataset/ISIC/ISIC2018_Task1-2_Training_Input/'
base_dir = "Dataset/ISIC2017/ISIC2017_Task1-2_Training_Input"
# Percorsi delle sottocartelle
folder1 = os.path.join(base_dir, '1')
folder2 = os.path.join(base_dir, '2')
folder3 = os.path.join(base_dir, '3')
training_folder = base_dir  # Immagini principali

# Ottieni i nomi dei file (solo immagini) nelle cartelle 1 e 2
def get_image_files(folder):
    return {f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}

images_in_folder2 = get_image_files(folder2)
images_in_folder3 = get_image_files(folder3)

# Unione dei file presenti nelle cartelle 1 e 2
all_images_in_2_and_3 = images_in_folder2.union(images_in_folder2)

# Ottieni i nomi dei file (solo immagini) nella cartella "Training"
images_in_training = get_image_files(training_folder)

# Verifica se l'immagine in "Training" non è presente in 1 e 2 e spostala in "3"
for image in images_in_training:
    if image not in all_images_in_2_and_3:
        src_path = os.path.join(training_folder, image)
        dest_path = os.path.join(folder1, image)  # Sposta direttamente in "3"
        shutil.move(src_path, dest_path)  # Sposta il file
        print(f"Spostato: {image}")

