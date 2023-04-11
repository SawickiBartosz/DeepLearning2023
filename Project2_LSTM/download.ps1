kaggle competitions download -c tensorflow-speech-recognition-challenge
Expand-Archive ./tensorflow-speech-recognition-challenge.zip

# unpack images using 7z
set-alias sz "$env:ProgramFiles\7-Zip\7z.exe"
sz x -otensorflow-speech-recognition-challenge tensorflow-speech-recognition-challenge\train.7z -r
sz x -otensorflow-speech-recognition-challenge tensorflow-speech-recognition-challenge\test.7z -r