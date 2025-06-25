import shutil
import os

if __name__ == "__main__":
    DIR = 'D:/Lab/data_pick/0606_total'
    fList = os.listdir(DIR)
    for fname in fList:
        if fname.endswith('.jpg'):
            new_name = fname[:-4]+'_fixed.jpg'
            old_path = os.path.join(DIR, fname)
            new_path = os.path.join(DIR, new_name)
            shutil.move(old_path, new_path)
            print(f"Renamed {fname} to {new_name}")
    