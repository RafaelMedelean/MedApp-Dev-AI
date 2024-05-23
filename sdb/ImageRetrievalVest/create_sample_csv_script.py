import os

if __name__ == "__main__":
    path_to_big_csv = "/workspaces/MedApp-Dev-AI/sdb2/all_patches_balanced_candidates.csv"
    path_to_augmented_jpegs = "/workspaces/MedApp-Dev-AI/sdb2/balanced_candidates_augmented_jpegs"
    path_to_new_csv = "/workspaces/MedApp-Dev-AI/sdb2/working_sample_jpegs.csv"

    if os.path.exists(path_to_new_csv):
        os.remove(path_to_new_csv)

    with open(path_to_big_csv, "r") as big_csv:
        lines = big_csv.readlines()
        with open(path_to_new_csv, "w") as new_csv:
            new_csv.write("numpy_filename,class_label,original_x,original_y,original_z,resampled_x,resampled_y,resampled_z\n")
            for line in lines:
                numpy_filename = line.split(",")[0]
                without_npy_extension = numpy_filename.replace(".npy", "")
                if(os.path.exists(path_to_augmented_jpegs + "/" + without_npy_extension)):
                    new_csv.write(line)
    



