
# download from kaggle API
kaggle competitions download -c cifar-10
# unzip archive
Expand-Archive .\cifar-10.zip
rm .\cifar-10.zip

# unpack images using 7z
set-alias sz "$env:ProgramFiles\7-Zip\7z.exe"
sz x -ocifar-10 cifar-10\train.7z -r
sz x -ocifar-10 cifar-10\test.7z -r

# make new directory for keras image_dataset_from_directory
mkdir cifar-10\train\train_images
Copy-Item cifar-10\train\*.png cifar-10\train\train_images
rm cifar-10\train\*.png
