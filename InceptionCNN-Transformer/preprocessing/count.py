# import json
# from collections import Counter

# # Load your dataset
# with open('/Users/lalith/code/projects/Sing_AR/archive/nslt_100.json', 'r') as f:
#     data = json.load(f)

# # Extract the class ID (the first number in the 'action' array) for every video
# class_counts = Counter(video_data['action'][0] for video_data in data.values())

# # Display the results sorted from fewest videos to most videos
# print("Class ID | Number of Videos")
# print("-" * 25)
# for class_id, count in sorted(class_counts.items(), key=lambda item: item[1]):
#     print(f"Class {class_id:<6} | {count} videos")

# print(f"\nTotal videos in this subset: {len(data)}")
# print(f"Total unique words (classes): {len(class_counts)}")


from pathlib import Path

def count_files_in_subdirectories(target_folder_path):
    base_path = Path("/Users/lalith/code/projects/Sing_AR/preprocessing/output")
    
    # Check if the path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: '{target_folder_path}' does not exist or is not a directory.")
        return

    print(f"{'Directory Name':<20} | {'Number of Files'}")
    print("-" * 40)
    
    # Iterate through all items in the base folder
    for item in base_path.iterdir():
        # Only process subdirectories
        if item.is_dir():
            # Count the number of files in this subdirectory
            # (ignoring any nested directories)
            file_count = sum(1 for f in item.iterdir() if f.is_file())
            
            print(f"{item.name:<20} | {file_count}")

# You can run it by passing the path to your output folder
output_folder_path = '/Users/lalith/code/projects/Sing_AR/output' 
count_files_in_subdirectories(output_folder_path)
