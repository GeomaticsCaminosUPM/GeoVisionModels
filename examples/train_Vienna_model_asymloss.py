base_dir = '/media/beegfs/home/u186/u186731/ParkingSpaceDetection/'

import sys # Add AsfSearch lib to path
sys.path.insert(1, base_dir)

import seg_utils as utils
import seg_losses as losses
import SemanticSegmentation as Seg
import geoimage_dataset as geoimage
import geometry_utils
import kornia

import lightning.pytorch as pl
import torch, torchmetrics
import numpy as np
import os
import segmentation_models_pytorch as smp
from seg_utils import SegmentationTransforms

np.random.seed(69)
torch.manual_seed(69)
torch.cuda.manual_seed(69)

dataset_name = "lb"
model_log_folder = base_dir + 'ViennaDataset/UnetSAM/Asymloss/' + dataset_name
dataset_folder = base_dir + 'ViennaDataset/res15cm_overlap15m'
img_dir = dataset_folder + "/" + dataset_name
mask_dir = dataset_folder + "/" + "GT"


encoder="sam-vit_b"
weights = "sa-1b" # input size must be 1024 for weights sa-1b
input_size = (1024, 1024) # input size must be square 
batch_size = 2
augment_rate = 4
num_workers = 1
background_index = 0

max_epochs = 10

mean=[0.44368124, 0.4537536, 0.45856014]
std=[0.1987232, 0.17570017, 0.15577587]

transforms = SegmentationTransforms.Compose([
    SegmentationTransforms.Resize(input_size),
    utils.aug_transforms_2,
    SegmentationTransforms.toTensor(),
    SegmentationTransforms.Normalize(mean,std),
    SegmentationTransforms.toTensor()
])

transforms_val = SegmentationTransforms.Compose([
    SegmentationTransforms.Resize(input_size),
    SegmentationTransforms.toTensor(),
    SegmentationTransforms.Normalize(mean,std),
    SegmentationTransforms.toTensor()
])

dm = Seg.get_Vienna_train_datamodule(img_dir = img_dir, mask_dir = mask_dir)

dm.preprocess(batch_size=batch_size,augment_rate=augment_rate,
              train_transforms=transforms,val_transforms=transforms_val,
              num_workers=num_workers, min_num_classes=2,min_area=0.001,balanced_inds=True)

class_weights = np.array(utils.get_class_weights(dm,balanced=True,method='area',normalize = True))

print(f"Dataset class weights {class_weights}")

gamma = 0.75
class_weights = np.power(class_weights, gamma)
class_weights = list(class_weights * len(class_weights) / np.sum(class_weights))

print(f"Loss class weights {class_weights}")

loss = losses.AsymmetricUnifiedFocalLoss(mu=0.5,delta=0.7, gamma=gamma, common_class_index=background_index,class_weights=class_weights)
accuracy = torchmetrics.JaccardIndex(task="multiclass", num_classes=dm.num_classes)

resolution = 0.15 # img resolution m
overlap = 15 # img overlap m

# ulabeled, road, tram, crosswalk, parking, private road, sidewalk, sep ped path
clean_mask_area = np.array([2,50,20,3,4,5,9,15]) # min area m^2
clean_mask_len =  np.array([0,1,0.75,0.5,1,0.5,0.5,0.5]) # min width m

clean_mask_area = clean_mask_area / (resolution**2) # img resolution
clean_mask_len = clean_mask_len / (2 * resolution) # 0.5 width and img resolution

def clean_mask_func(mask):
    return utils.clean_mask_by_area(mask,clean_mask_area,clean_mask_len,background_index=background_index,open_first=True,footprint="octagon")

def unetSAM(num_classes,input_size, encoder,weights=None,freeze_encoder=True):
    import seg_utils as utils
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_encoder
    if weights == None:
        if freeze_encoder:
            import warnings
            warnings.warn("Setting freeze_encoder to False as no weights were given")
            freeze_encoder = False
    _encoder = get_encoder(encoder, weights=weights, img_size=input_size[0], depth=4,in_channels=3)
    model=smp.create_model("unetplusplus", encoder,encoder_weights=weights, encoder_depth=4, decoder_channels=[256, 128, 64, 32],in_channels=3,classes=num_classes)
    model.encoder = _encoder
    if freeze_encoder:
        utils.set_parameter_requires_grad(model.encoder,False)
    return model

model_arch = unetSAM(dm.num_classes,input_size, encoder,weights,True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"device: {device}")

epoch = 0
model = Seg.Model(dm.num_classes,model=model_arch,loss = loss, learning_rate=0.0001,
                  accuracy=accuracy,device=device)

utils.count_parameters(model)

torch.set_float32_matmul_precision('medium')
print(f"CUDA available: {torch.cuda.is_available()}")

trainer = pl.Trainer(
    max_epochs=max_epochs - epoch,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else num_workers,
    default_root_dir=model_log_folder,
    enable_checkpointing=True
)

trainer.fit(model, dm)

trainer.test(model, datamodule=dm)

max_version = -1
max_version_folder = None

for folder in os.listdir(model_log_folder+"/lightning_logs"):
    version_number = Seg.get_version_number(folder)
    if version_number > max_version:
        max_version = version_number
        max_version_folder = folder

checkpoint_folder = os.path.normpath(model_log_folder + "/lightning_logs/" + max_version_folder)


test_dm = Seg.get_Vienna_test_datamodule(img_dir=img_dir,test_folders="val")
test_dm.preprocess(batch_size=1,augment_rate=1,test_transforms=transforms_val,num_workers=num_workers)
test_out_folder = checkpoint_folder + "/model_test/" + folder
Seg.geoimage_test(test_out_folder,test_dm,model,clean_mask_func,overlap=overlap)


