from google.colab import files
uploaded = files.upload()

# This will show you the exact name Colab saved the file as
for name in uploaded.keys():
  print(f'User uploaded file "{name}"')
