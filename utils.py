import os
import shutil

def flatten_folder(parent_dir, nested_name):
    nested_dir = os.path.join(parent_dir, nested_name)
    
    if not os.path.exists(nested_dir):
        print(f"⚠️ Skipping {nested_dir} (not found)")
        return

    for file in os.listdir(nested_dir):
        src = os.path.join(nested_dir, file)
        dst = os.path.join(parent_dir, file)

        shutil.move(src, dst)

    os.rmdir(nested_dir)
    print(f"✅ Flattened {nested_name} into {parent_dir}")

# Apply to train and valid
flatten_folder("train", "DIV2K_train_HR")
flatten_folder("valid", "DIV2K_valid_HR")

print("🎉 All done!")