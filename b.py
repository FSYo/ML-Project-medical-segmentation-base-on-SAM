import SimpleITK as sitk
import matplotlib.pyplot as plt

# 读取 NIfTI 文件
nifti_img = sitk.ReadImage('Training/img/img0005.nii.gz')

# 获取图像数组
img_array = sitk.GetArrayFromImage(nifti_img)

# 显示切片
plt.imshow(img_array[:, :, img_array.shape[2]//2], cmap='gray')
plt.show()
