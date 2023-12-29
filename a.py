import os
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_and_save_image_label(image_file_path, label_file_path, output_folder):
    try:
        # 读取NIfTI文件
        img = nib.load(image_file_path)
        label_img = nib.load(label_file_path)
        
        # 获取图像和标签数据
        image_data = img.get_fdata()
        label_data = label_img.get_fdata()

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 显示图像和标签图像切片，并保存到文件夹（这里只显示第一个切片，你可以根据需要进行修改）
        num_slices = image_data.shape[-1]
        print(num_slices)
        for i in range(99, 100):
            plt.subplot(1, 2, 1)
            plt.imshow(image_data[:, :, i], cmap='gray')
            plt.title(f'Image Slice {i + 1}')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            h, w = label_data[:, :, i].shape
            for j in range(h):
                for k in range(w) : 
                    print(label_data[j][k][i], end = ' ')
                print()
            plt.imshow(label_data[:, :, i], cmap='jet')  # 使用 'jet' 颜色映射，你可以根据需要调整
            plt.title(f'Label Slice {i + 1}')
            plt.colorbar()

            # 保存图像到文件夹
            save_path = os.path.join(output_folder, f'slice_{i + 1}.png')
            plt.savefig(save_path)
            plt.close()

    except Exception as e:
        print(f"发生错误：{e}")

# 指定NIfTI文件路径和输出文件夹
image_file_path = "Training/img/img0001.nii.gz"
label_file_path = "Training/label/label0001.nii.gz"
output_folder = "1"  # 更改为你想要保存结果的文件夹路径

# 可视化图像和标签NIfTI文件，并保存到文件夹
visualize_and_save_image_label(image_file_path, label_file_path, output_folder)
