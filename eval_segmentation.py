import torch

def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = torch.zeros([self.num_class,self.num_class]).cuda(0)

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    def recall_accuracy(self):
        acc = torch.zeros(self.num_class).cuda()
        for class_i in range(0,self.num_class):
            acc[class_i] = self.confusion_matrix[class_i,class_i]/self.confusion_matrix[class_i].sum()
        return acc
    def precision(self):
        acc = torch.zeros(self.num_class).cuda()
        for class_i in range(0,self.num_class):
            acc[class_i] = self.confusion_matrix[class_i,class_i]/torch.sum(self.confusion_matrix[:,class_i])
        return acc
    def Mean_Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        Acc = torch_nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                    torch.diag(self.confusion_matrix))
        MIoU = torch_nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, dim=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                    torch.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # print(gt_image[mask].int().device)
        # print(pre_image[mask].device)
        label = self.num_class * gt_image[mask].int() + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        gt_image = gt_image.cuda(0)
        pre_image = pre_image.cuda(0)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros([self.num_class, self.num_class]).cuda()