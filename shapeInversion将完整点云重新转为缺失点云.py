import torch
import numpy
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw
def k_mask( target, x, stage=-1):
    """
    masking based on CD.
    target: (1, N, 3), partial, can be < 2048, 2048, > 2048
    x: (1, 2048, 3)
    x_map: (1, N', 3), N' < 2048
    x_map: v1: 2048, 0 masked points
    """
    stage = max(0, stage)
    masking_option="element_product"
    k_mask_k=[5,5,5,5]
    knn = k_mask_k[stage]
    if knn == 1:
        cd1, cd2, argmin1, argmin2 = distChamfer(target, x)
        idx = torch.unique(argmin1).type(torch.long)
    elif knn > 1:
        # dist_mat shape (B, 2048, 2048), where B = 1
        dist_mat = distChamfer_raw(target, x)
        # indices (B, 2048, k)
        val, indices = torch.topk(dist_mat, k=knn, dim=2, largest=False)
        # union of all the indices
        idx = torch.unique(indices).type(torch.long)

    if masking_option == 'element_product':
        mask_tensor = torch.zeros(2048, 1)
        mask_tensor[idx] = 1
        mask_tensor = mask_tensor.unsqueeze(0)
        x_map = torch.mul(x, mask_tensor)
    elif masking_option == 'indexing':
        x_map = x[:, idx]

    return x_map
crop=[]
completion=[]
with open("./crop_txt1.txt", 'r') as f:

    for i in f:
        temp = []
        temp.append(float(i.strip().split(',')[0]))
        temp.append(float(i.strip().split(',')[1]))
        temp.append(float(i.strip().split(',')[2]))
        # print(temp)
        crop.append(temp)
with open("./real_txt1.txt", 'r') as f:
    for i in f:
        temp = []
        temp.append(float(i.strip().split(',')[0] ))
        temp.append(float(i.strip().split(',')[1] ))
        temp.append(float(i.strip().split(',')[2]))
        completion.append(temp)
# print(crop)

crop=numpy.array(crop)
completion=numpy.array(completion)
completion=torch.from_numpy(completion)
crop=torch.from_numpy(crop)
crop=torch.unsqueeze(crop,0)
completion=torch.unsqueeze(completion,0)
print(crop.shape)
print(completion.shape)

a=k_mask(crop,completion)
a=torch.squeeze(a,0).numpy()

numpy.savetxt('./inversion_txt'+'.txt', a, fmt = "%f,%f,%f")

print(a)