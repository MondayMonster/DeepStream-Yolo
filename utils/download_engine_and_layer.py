import gdown
import os
import zipfile

# Create directories if they do not exist
os.makedirs(os.path.expanduser('~/Documents/DeepStream-Yolo'), exist_ok=True)
#os.makedirs('/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream', exist_ok=True)

# Download the first file to the specified directory
file_id_1 = '1ZEQiBXLsT-48EVfAh_ctt-P5zh6Pilki'
output_path_1 = os.path.join(os.path.expanduser('~/Documents/DeepStream-Yolo'), 'maritimo.engine')
download_url_1 = f'https://drive.google.com/uc?id={file_id_1}'
gdown.download(download_url_1, output_path_1, quiet=False)
print(f"Downloaded maritimo.engine to {output_path_1}")

# Download the zip file to the specified directory
#file_id_2 = '1WBUzy72mjtYBNp5KINbi8p9c_r3Zwzus'
#zip_output_path = os.path.join('/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream', 'nvdsinfer_custom_impl_Yolo.zip')
#download_url_2 = f'https://drive.google.com/uc?id={file_id_2}'
#gdown.download(download_url_2, zip_output_path, quiet=False)
#print(f"Downloaded nvdsinfer_custom_impl_Yolo.zip to {zip_output_path}")

# Unzip the file into the specified folder
#with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
#    extract_path = '/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps-1.1.6/apps/deepstream-imagedata-multistream/nvdsinfer_custom_impl_Yolo'
#    zip_ref.extractall(extract_path)
#    print(f"Extracted nvdsinfer_custom_impl_Yolo to {extract_path}")
