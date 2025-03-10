import lightning.pytorch as pl
from torch.utils.data import Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


def get_version_number(folder_name):
    import re
    match = re.search(r'version_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return 0

def get_epoch_and_step(filename):
    import re
    epoch_match = re.search(r'epoch=(\d+)', filename)
    step_match = re.search(r'step=(\d+)', filename)
    if epoch_match and step_match:
        epoch = int(epoch_match.group(1))
        step = int(step_match.group(1))
        return epoch, step
    return None, None

def get_checkpoint_folder(model_log_folder, remove=True):
    import os, shutil
    max_version = -1
    max_version_folder = None
    if not os.path.isdir(model_log_folder+"/lightning_logs"):
        raise Exception(f"Folder {model_log_folder+'/lightning_logs'} not found")
    
    for folder in os.listdir(model_log_folder+"/lightning_logs"):
        version_number = get_version_number(folder)
        if version_number > max_version:
            max_version = version_number
            max_version_folder = folder

    if max_version == -1:
        raise Exception(f"Folder not found {model_log_folder+'/lightning_logs'}")
    
    checkpoint_folder = os.path.normpath(model_log_folder+ "/lightning_logs/" + max_version_folder + "/checkpoints")
    if not os.path.isdir(checkpoint_folder):
        if remove:
            rmfolder = os.path.normpath(model_log_folder+ "/lightning_logs/" + max_version_folder)
            print(f"Warning. Deleting folder with no checkpoint {rmfolder}.")
            shutil.rmtree(rmfolder, ignore_errors=True)
            return get_checkpoint_folder(model_log_folder)
        else:
            raise Exception(f"Folder not found {checkpoint_folder}.")
    
    return checkpoint_folder, max_version

def get_best_epoch(checkpoint_folder):
    import os
    best_epoch = -1
    best_step = -1
    best_checkpoint = None
    for filename in os.listdir(checkpoint_folder):
        if os.path.isfile(os.path.join(checkpoint_folder, filename)):
            epoch, step = get_epoch_and_step(filename)
            if epoch is not None and step is not None:
                if epoch > best_epoch or (epoch == best_epoch and step > best_step):
                    best_epoch = epoch
                    best_step = step
                    best_checkpoint_file = filename

    if best_epoch == -1:
        raise Exception(f"No epoch=x-step=y.ckpt file found in {checkpoint_folder}")
    
    return best_checkpoint_file, best_epoch

def load_from_checkpoint(model_log_folder,model,loss=None,accuracy=None,remove=True):
    checkpoint_folder, max_version = get_checkpoint_folder(model_log_folder,remove=remove)
    best_checkpoint_file, best_epoch = get_best_epoch(checkpoint_folder)
    return Model.load_from_checkpoint(checkpoint_folder + "/" + best_checkpoint_file,
                               model=model,accuracy=accuracy,loss=loss), best_epoch


def geoimage_test(output_folder:str,test_datamodule,model,clean_func=None, overlap = 0, print_progress:int = 10):
    import geometry_utils
    import geoimage_dataset as geoimage 
    import os 
    import warnings

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        print(f"output folder {output_folder} created")

    test_dataloader = test_datamodule.test_dataloader()
    test_image_paths = test_datamodule.test_paths
    i=0
    for x in iter(test_dataloader):
        if type(x) is list and len(x) == 2:
            x = x[0]

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            crs = geometry_utils.get_crs_of_img(test_image_paths[i])
            bounds = geometry_utils.get_bounds_of_img(test_image_paths[i],crs=crs)

        if overlap > 0:
            bounds_no_overlap = bounds.to_crs(bounds.estimate_utm_crs(datum_name='WGS 84'))
            bounds_no_overlap = bounds_no_overlap.buffer(-overlap)

        pred = model.predict(x,get_prob=False)

        for j in pred:
            if clean_func:
                y = clean_func(j)
            else:
                y = j

            if overlap > 0:
                minx,miny,maxx,maxy = bounds.to_crs(bounds.estimate_utm_crs(datum_name='WGS 84')).total_bounds
                pixx = round(overlap / ((maxx - minx) / y.shape[0])) 
                pixy = round(overlap / ((maxy - miny) / y.shape[1])) 
                y = y[pixx:-pixx,pixy:-pixy]
                bounds = bounds_no_overlap

            name = output_folder + "/" + test_image_paths[i].split("/")[-1]
            name = name.replace(".jpg", ".png")
            name = name.replace("image","mask")

            geoimage.save_geoarray(y,name,bounds=bounds,driver="PNG")

            if print_progress == 0 or print_progress == 1:
                print(f"model output saved as {name}. {round(i*100/len(test_image_paths),ndigits=2)} % done")
            elif print_progress > 0 and i % print_progress == 0:
                print(f"{i} images tested. {round(i*100/len(test_image_paths),ndigits=2)} % done")

            i += 1

    print(f"Test finished. Outputs can be found on {output_folder}")


class Model(pl.LightningModule):
    def __init__(self, num_classes, model, accuracy=None, loss=None, learning_rate = 2e-4, device=None):
        super().__init__()
        import copy, torch

        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.accuracy = copy.deepcopy(accuracy)
        self.loss = copy.deepcopy(loss)

        self.model = model
        self.devices = device
        self.class_weights = None

        try:
            self.loss.weight = torch.tensor(self.loss.weight).float()
            if device:
                self.loss.weight = self.loss.weight.to(device=device)

            self.class_weights = self.loss.weight
        except:
            try:
                self.loss.weights = torch.tensor(self.loss.weights).float()
                if device:
                    self.loss.weights = self.loss.weights.to(device=device)

                self.class_weights = self.loss.weights
            except:
                try:
                    self.loss.class_weights = torch.tensor(self.loss.class_weights).float()
                    if device:
                        self.loss.class_weights = self.loss.class_weights.to(device=device)

                    self.class_weights = self.loss.class_weights
                except:
                    try: 
                        self.loss.to_device(device)
                    except:
                        None

        if type(self.class_weights) != type(None):
            if len(self.class_weights) != num_classes:
                print(f"Warning. len loss weights is {len(self.class_weights)} but num classes is {num_classes}. Errors might occur.")

        self.save_hyperparameters(ignore=['model','accuracy','loss'])
        
    def forward(self, x):
        import torch.nn.functional as F
        x = x.to(device=self.device) 
        x = self.model(x)
        try:
            x = x['out']
        except:
            None

        return x
    
    def run(self,x):
        import torch
        y = self.forward(x)
        return torch.Tensor.cpu(y)        

    def training_step(self, batch):
        import torch.nn.functional as F
        x, y = batch
        logits = self.forward(x)
        l = self.loss(logits, y)
        return l
    
    def test_step(self, batch, batch_idx):
        import torch.nn.functional as F
        import torch
        x, y = batch
        logits = self.forward(x)
        l = self.loss(logits, y)
        preds = torch.argmax(logits,dim=1)#F.softmax(logits,dim=1), dim=1)
        acc = self.accuracy(preds, y)
        metrics = {"test_acc": acc, "test_loss": l}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        import torch.nn.functional as F
        import torch
        x, y = batch
        logits = self.forward(x)
        l = self.loss(logits, y)
        preds = torch.argmax(logits,dim=1)#F.softmax(logits,dim=1), dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", l, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def predict(self,x,get_prob=False):
        import torch.nn.functional as F
        import torch

        logits = self.forward(x)
        pred = torch.argmax(logits, dim=1).cpu()

        #pred_numpy = []
        _pred = pred.detach().numpy().astype('uint8')
        pred_numpy = [_pred[i] for i in range(pred.shape[0])]
        #for i in range(pred.shape[0]):
        #    pred_numpy.append(_pred[i,:,:])
        
        #if len(pred_numpy) == 1:
        #    pred_numpy = pred_numpy[0]

        if get_prob:
            probabilities = F.softmax(logits, dim=1)
            prob = probabilities[:,pred,:,:].cpu()
            _prob = prob.detach().numpy().astype('float32')
            prob_numpy = [_prob[i] for i in range(prob.shape[0])]

            #if len(prob_numpy) == 1:
            #    prob_numpy = prob_numpy[0]

            return pred_numpy, prob_numpy
        else:
            return pred_numpy
    
    def configure_optimizers(self):
        import torch
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
# Define a pytorch dataset
class SegDataset(Dataset):
    def __init__(self, input_paths, label_dict, transform=None, input_transform = None, 
                 target_transform = None, istest:bool=False, translate_labels = None):
        
        self.input_paths = input_paths
        self.label_dict = label_dict
        self.transform = transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.istest = istest
        self.translate_labels = translate_labels

        not_found = 0
        for i in self.input_paths:
            try:
                self.label_dict[i]
            except:
                if istest == False:
                    raise Exception(f"Key {i} not found in label_dict")
                else:
                    not_found += 1 

        if not_found == 0:
            self.istest = False

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        # Load data and get label

        if self.istest:
            return self.path_to_torch(index,None)
        else:
            return self.path_to_torch(index,index)
    
    def data_to_torch(self,X,Y=None):
        import torch
        import torchvision

        if self.input_transform:
            X = self.input_transform(X)

        if self.target_transform and type(Y) != type(None):
            Y = self.target_transform(Y)

        if self.transform and type(Y) != type(None):
            X,Y = self.transform(X,Y)
        else:
            # There could be errors as X is inputed as target too
            X,_ = self.transform(X,X)
        

        if type(Y) is torch.Tensor or type(Y) is torchvision.tv_tensors.Image:
            Y = torch.squeeze(Y)
            if ((Y > 0) * (Y < 1)).any(): #any(Y[Y > 0] < 1):
                Y = Y * 255

            Y = Y.type(torch.LongTensor)   

        if type(Y) != type(None):
            return X,Y 
        else:
            return X
    
    def path_to_torch(self,X,Y=None):
        from PIL import Image

        if type(X) is int:
            _X = Image.open(self.input_paths[X]) 
            if type(Y) == type(None) and self.istest == False:
               _Y = Image.open(self.label_dict[self.input_paths[X]])      
        else:
            _X = Image.open(X)

        if type(Y) is int:
            _Y = Image.open(self.label_dict[self.input_paths[Y]])
        elif type(Y) != type(None):
            _Y = Image.open(Y)

        if type(Y) != type(None) and type(self.translate_labels) != type(None):
            _Y = self.translate(_Y)

        if type(Y) != type(None):
            return self.data_to_torch(_X,_Y) 
        else:
            return self.data_to_torch(_X)
        
    def translate(self,Y): 
        import seg_utils 
        return seg_utils.translate_mask(Y,self.translate_labels,values = None)
    

    

#from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
class SegDataModule(pl.LightningDataModule):
    def __init__(self, num_classes:int, train_paths:list = [], train_mask_dict:dict = {}, val_paths:list = [], val_mask_dict:dict = {},
                  test_paths:list = [], test_mask_dict:dict = {},
                  ignore_index = None, labels=None, rgb_to_labels=None, translate_labels = None):
        super().__init__()
        import numpy as np

        istest = False 
        if len(train_paths) == 0 and len(val_paths) == 0:
            istest = True
            print("Datamodule only for testing model")

        self.train_paths = train_paths

        self.val_paths = val_paths

        self.test_paths = test_paths

        self.mask_dict = {**train_mask_dict, ** val_mask_dict, ** test_mask_dict}

        if len(self.train_paths) == 0 and istest == False:
            print("Warning. train_paths is empty")

        self.num_classes = num_classes

        self.ignore_index = ignore_index

        self.labels = labels 

        if labels:
            l = list(labels.keys())
            if len(l) != self.num_classes:
                print(f"Warning. length of labels is {len(l)} and num_classes is {self.num_classes}. Labels are {l}")
                self.num_classes = len(l)
                print(f"Setting num_classes to {len(l)}.")

        self.rgb_to_labels = rgb_to_labels

        #if type(translate_labels) != type(None):
        #    for i in range(num_classes):
        #        if i not in translate_labels.keys():
        #            translate_labels[i] = i 

        self.translate_labels = translate_labels

        self.input_paths = list(np.unique([*self.train_paths,*self.val_paths,*self.test_paths]))

        if len(self.mask_dict.keys()) != len(self.input_paths) and istest == False:
            print(f"Warning. length of mask_dict keys is {len(self.mask_dict.keys())} and length of (all) input_paths is {len(self.input_paths)}")

        self.preprocess_done = False
        self.istest = istest
        self.allow_empty_masks = None

    def preprocess(self,train_transforms=None,val_transforms=None,test_transforms=None,
                    augment_rate=0,batch_size:int=1,num_workers:int=0,
                    train_perc:float=0.8, test_perc:float = 0, min_num_classes:int=0, min_area:float = 0, 
                    balanced_inds:bool = False, get_countmatrix:bool = True):
        
        from sklearn.model_selection import train_test_split
        import copy

        if self.preprocess_done:
            raise Exception("You must create the datamodule again to change any parameter")

        self.min_num_classes = min_num_classes
        self.batch_size = batch_size
        if augment_rate <= 0:
            augment_rate = 1
        elif augment_rate < 1:
            balanced_inds = True  

        self.augment_rate = augment_rate
        self.num_workers = num_workers
        if len(self.val_paths) == 0 and len(self.train_paths) > 0:
            test_perc = 1 - train_perc
            self.train_paths, self.val_paths = train_test_split(self.train_paths,test_size=test_perc,train_size=train_perc,random_state=69,shuffle=True)

        if len(self.test_paths) == 0:
            if test_perc > 0:
                val_perc = 1 - test_perc
                self.val_paths, self.test_paths = train_test_split(self.train_paths,test_size=test_perc,train_size=val_perc,random_state=69,shuffle=True)
            else:
                print("Warning. No test paths given and test_perc = 0 so test will be the same a validation")
                self.test_paths = copy.copy(self.val_paths)     

        if type(test_transforms) == type(None):
            test_transforms = copy.copy(val_transforms)

     
        self.train_transforms = copy.copy(train_transforms)
        self.val_transforms = copy.copy(val_transforms)
        self.test_transforms = copy.copy(test_transforms)

        if get_countmatrix or balanced_inds or min_num_classes > 1:
            import seg_utils
            import numpy as np

            self2 = copy.deepcopy(self)
            self2.preprocess_done = True

            self.countmatrix = seg_utils.get_countmatrix(self2.train_dataloader_all(),num_classes=self.num_classes)

            if min_num_classes > 1:
                inds, self.countmatrix = seg_utils.delete_empty_masks(countmatrix=self.countmatrix,min_num_classes=min_num_classes, min_area = min_area)
                print(f"{len(self.train_paths) - len(inds)} empty masks deleted")
                self.train_paths_augmented = list(np.array(self.train_paths)[inds])
            else:
                self.train_paths_augmented = copy.copy(self.train_paths)

            if balanced_inds:
                _balanced_inds = seg_utils.get_balanced_inds(countmatrix=self.countmatrix,n=int(round(self.countmatrix.shape[0] * self.augment_rate)))
                self.train_paths_augmented = list(np.array(self.train_paths_augmented)[_balanced_inds])
        else:
            self.train_paths_augmented = copy.copy(self.train_paths)

        if not balanced_inds:
            if augment_rate > 1:
                for _ in range(self.augment_rate-1):
                    self.train_paths_augmented = [*self.train_paths_augmented,*self.train_paths_augmented]

        self.preprocess_done = True

    def train_dataloader_all(self) -> TRAIN_DATALOADERS:
        from torch.utils.data import DataLoader
        import copy

        if self.preprocess_done == False:
            raise Exception("Run self.preprocess() before training")
        
        p = copy.deepcopy(self.train_paths)

        train_dataset_clean = SegDataset(p, self.mask_dict, transform=self.train_transforms,istest=self.istest, translate_labels=self.translate_labels)
        return DataLoader(train_dataset_clean, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        from torch.utils.data import DataLoader
        import copy

        if self.preprocess_done == False:
            raise Exception("Run self.preprocess() before training")
        
        p = copy.deepcopy(self.train_paths_augmented)

        train_dataset = SegDataset(p, self.mask_dict, transform=self.train_transforms,istest=self.istest, translate_labels=self.translate_labels)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    

    def val_dataloader(self) -> EVAL_DATALOADERS:
        from torch.utils.data import DataLoader

        if self.preprocess_done == False:
            raise Exception("Run self.preprocess() before validation")
        

        val_dataset = SegDataset(self.val_paths, self.mask_dict, transform=self.val_transforms,istest=self.istest, translate_labels=self.translate_labels)
        if self.num_workers > 0:
            pers = True
        else:
            pers = False

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=pers)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        from torch.utils.data import DataLoader

        if self.preprocess_done == False:
            raise Exception("Run self.preprocess() before testing")
        

        test_dataset = SegDataset(self.test_paths, self.mask_dict, transform=self.test_transforms,istest=self.istest, translate_labels=self.translate_labels)
        if self.num_workers > 0:
            pers = True
        else:
            pers = False

        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=pers)
    
    
    def all_paths(self):
        import numpy as np
        return list(np.unique([*self.train_paths,*self.val_paths,*self.test_paths]))
    

def get_img_paths(folder:str, find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    import os
    image_paths = []
    for i in os.scandir(folder):
        if i.is_file():
            add = True 
            for j in find_criteria.keys():
                if (str(j) in str(i.name)) != find_criteria[j]:
                    add = False 

            if add: 
                image_paths.append(os.path.normpath(folder + "/" + str(i.name)))
    
    return image_paths

def get_mask_dict(image_paths:list, 
                  mask_folder = None, img_to_mask:dict = {"image":"mask",".jpg":".png"},
                  mask_dict:dict = dict()):
    
    import ntpath, os
    _image_paths = []
    for i in range(len(image_paths)):
        img_folder, mask = ntpath.split(image_paths[i])
        if type(mask_folder) == type(None):
            _mask_folder = img_folder 
        else:
            _mask_folder = mask_folder 

        for j in img_to_mask.keys():
            mask = mask.replace(str(j),img_to_mask[j])

        mask = os.path.normpath(_mask_folder + "/" + mask)
        if os.path.isfile(mask):
            mask_dict[image_paths[i]] = mask
            _image_paths.append(image_paths[i])
        else:
            print(f"Warning. Mask file {mask} not found for image {image_paths[i]}")
    
    return _image_paths, mask_dict


def get_data(img_folder:str,mask_folder = None,
             find_criteria:dict = {"image":True,".jpg":True,".aux":False}, img_to_mask:dict = {"image":"mask",".jpg":".png"}, 
             mask_dict = dict()):
    
    return get_mask_dict(get_img_paths(img_folder, find_criteria=find_criteria), 
                         mask_folder=mask_folder, img_to_mask=img_to_mask, mask_dict=mask_dict)

def get_train_datamodule(labels:dict, rgb_to_labels:dict, ignore_index:int, num_classes:int, translate_labels = None, 
                        img_dir:str = "", mask_dir = None,  
                        train_folders = "", val_folders = None, img_folder = "", mask_folder = "",
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
                        img_to_mask:dict = {"image":"mask",".jpg":".png"}):
    
    import os
    import numpy as np
    
    if type(train_folders) is str:
        train_folders = [train_folders]
    
    if type(val_folders) is str:
        val_folders = [val_folders]

    if not os.path.isdir(img_dir):
        raise Exception(f"img_dir {img_dir} does not exist")
    
    if type(mask_dir) == type(None):
        mask_dir = img_dir 
    elif not os.path.isdir(mask_dir):
        raise Exception(f"mask_dir {mask_dir} does not exist")
    
    if type(train_folders) is not list:
        raise Exception("train_folders should be a list of paths")
    
    if type(val_folders) is not list and type(val_folders) is not type(None):
        raise Exception("val_folders should be a list of paths")
    
    train_paths = []
    mask_dict = {}
    for i in train_folders:
        img_path = os.path.normpath(img_dir + "/" + i + "/" + img_folder)
        mask_path = os.path.normpath(mask_dir + "/" + i + "/" + mask_folder)
        new_train_paths = get_img_paths(img_path,find_criteria=find_criteria)
        new_train_paths, mask_dict = get_mask_dict(new_train_paths, mask_folder=mask_path, img_to_mask=img_to_mask, mask_dict=mask_dict)
        train_paths += new_train_paths

    val_paths = []
    if type(val_folders) is not type(None):
        for i in val_folders:
            img_path = os.path.normpath(img_dir + "/" + i + "/" + img_folder)
            mask_path = os.path.normpath(mask_dir + "/" + i + "/" + mask_folder)
            new_val_paths = get_img_paths(img_path,find_criteria=find_criteria)
            new_val_paths, mask_dict = get_mask_dict(new_val_paths, mask_folder=mask_path, img_to_mask=img_to_mask, mask_dict=mask_dict)
            val_paths += new_val_paths

    dm = SegDataModule(num_classes=num_classes,train_paths=train_paths,train_mask_dict=mask_dict, 
                         val_paths=val_paths,ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels, translate_labels=translate_labels)
    
    return dm

def get_test_datamodule(labels:dict, rgb_to_labels:dict, num_classes:int, ignore_index:int = -1, translate_labels = None, 
                        img_dir:str = "", test_folders = None, img_folder = "", 
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    
    import os
    
    if type(test_folders) is str:
        test_folders = [test_folders]

    if type(test_folders) is not list:
        raise Exception("test_folders should be a list of paths")
    
    test_paths = []
    #mask_dict = {}
    for i in test_folders:
        img_path = os.path.normpath(img_dir + "/" + i + "/" + img_folder)
        #mask_path = os.path.normpath(mask_dir + "/" + i + "/" + mask_folder)
        new_test_paths = get_img_paths(img_path,find_criteria=find_criteria)
        #new_test_paths, mask_dict = get_mask_dict(new_test_paths, mask_folder=mask_path, img_to_mask=img_to_mask, mask_dict=mask_dict)
        test_paths += new_test_paths

    return SegDataModule(num_classes=num_classes,test_paths=test_paths,
                         ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels, translate_labels = translate_labels)
    

def get_parking_train_datamodule(img_dir:str, mask_dir = None,  
                        train_folders = "", val_folders = "", img_folder = "img", mask_folder = "mask",
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
                        img_to_mask:dict = {"image":"mask",".jpg":".png"}):

    num_classes = 2
    ignore_index = None
    labels = {
        0 : {'class' : 'parking_space', 'rgb' : [255,255,255]},
        1 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '255255255' : 0,
        '000000000' : 1
    }

    return get_train_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                                num_classes=num_classes, img_dir=img_dir, mask_dir=mask_dir, 
                                train_folders=train_folders,val_folders=val_folders, 
                                find_criteria=find_criteria, img_to_mask=img_to_mask,
                                img_folder=img_folder,mask_folder=mask_folder)

def get_parking_test_datamodule(img_dir:str,test_folders="",img_folder="img",
                                find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    num_classes = 2
    ignore_index = None
    labels = {
        0 : {'class' : 'parking_space', 'rgb' : [255,255,255]},
        1 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '255255255' : 0,
        '000000000' : 1
    }

    return get_test_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                                num_classes=num_classes,img_dir=img_dir,test_folders=test_folders,
                                img_folder=img_folder, find_criteria=find_criteria)

def desired_classes_to_dict(labels,rgb_to_labels,desired_classes,background_index,num_classes):
    import numpy as np
    if type(desired_classes) != type(None):
        new_labels = {}
        new_rgb = {}
        rgb_k = np.array(list(rgb_to_labels.keys()))
        rgb_v = np.array(list(rgb_to_labels.values()))
        translate_labels = {}
        i = 1 
        for c in labels.keys():
            if c == background_index:
                new_labels[c] = labels[c]
                new_rgb[rgb_k[rgb_v == c][0]] = c
                continue
            else:
                if c in desired_classes:
                    translate_labels[c] = i
                    new_labels[i] = labels[c]
                    new_rgb[rgb_k[rgb_v == c][0]] = i
                    i += 1 
                else:
                    translate_labels[c] = background_index

        rgb_to_labels = new_rgb
        labels = new_labels
        num_classes_new = len(np.unique(np.array(list(translate_labels.values()) + [background_index])))
    else:
        num_classes_new = num_classes
        translate_labels = None

    return labels, rgb_to_labels, translate_labels, num_classes_new

def get_Madrid_train_datamodule(desired_classes = None, img_dir:str = "", mask_dir = None,    
                        train_folders = "", val_folders = "", img_folder = "img", mask_folder = "mask",
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
                        img_to_mask:dict = {"image":"mask",".jpg":".png"}):
    
    num_classes = 7
    ignore_index = None
    background_index = 0
    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'building', 'rgb' : [150,0,0]},
        2 : {'class' : 'street', 'rgb' : [150,150,150]},
        3 : {'class' : 'sidewalk', 'rgb' : [150,0,150]},
        4 : {'class' : 'pool', 'rgb' : [0,150,255]},
        5 : {'class' : 'bike_lane', 'rgb' : [255,0,0]},
        6 : {'class' : 'parking', 'rgb' : [255,150,0]},
    }

    rgb_to_labels = {
        '000000000' : 0,
        '150000000' : 1,
        '150150150' : 2,
        '150000150' : 3,
        '000150255' : 4,
        '255000000' : 5,
        '255150000' : 6
    }
    
    labels, rgb_to_labels, translate_labels, num_classes = desired_classes_to_dict(labels,rgb_to_labels,desired_classes,background_index, num_classes)
        

    return get_train_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,translate_labels=translate_labels,
                                ignore_index=ignore_index,
                                num_classes=num_classes, img_dir=img_dir, mask_dir=mask_dir, 
                                train_folders=train_folders,val_folders=val_folders, 
                                find_criteria=find_criteria, img_to_mask=img_to_mask,
                                img_folder=img_folder,mask_folder=mask_folder)

def get_Madrid_test_datamodule(desired_classes = None, img_dir:str = "",test_folders="",img_folder="img", 
                                find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    
    num_classes = 7
    ignore_index = None
    background_index = 0
    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'building', 'rgb' : [150,0,0]},
        2 : {'class' : 'street', 'rgb' : [150,150,150]},
        3 : {'class' : 'sidewalk', 'rgb' : [150,0,150]},
        4 : {'class' : 'pool', 'rgb' : [0,150,255]},
        5 : {'class' : 'bike_lane', 'rgb' : [255,0,0]},
        6 : {'class' : 'parking', 'rgb' : [255,150,0]},
    }

    rgb_to_labels = {
        '000000000' : 0,
        '150000000' : 1,
        '150150150' : 2,
        '150000150' : 3,
        '000150255' : 4,
        '255000000' : 5,
        '255150000' : 6
    }
    
    labels, rgb_to_labels, translate_labels, num_classes = desired_classes_to_dict(labels,rgb_to_labels,desired_classes,background_index, num_classes)
        
    return get_test_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,translate_labels=translate_labels,ignore_index=ignore_index,
                                num_classes=num_classes,img_dir=img_dir,test_folders=test_folders,
                                img_folder=img_folder, find_criteria=find_criteria)


def get_Vienna_train_datamodule(desired_classes = None, img_dir:str = "", mask_dir = None,    
                        train_folders = "train", val_folders = "val", img_folder = "img", mask_folder = "mask",
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
                        img_to_mask:dict = {"image":"mask",".jpg":".png"}):
    
    num_classes = 8
    ignore_index = None
    background_index = 0

    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'Road', 'rgb' : [150,0,0]},
        2 : {'class' : 'Tram', 'rgb' : [150,150,150]},
        3 : {'class' : 'Crosswalk', 'rgb' : [150,0,150]},
        4 : {'class' : 'Parking', 'rgb' : [0,150,255]},
        5 : {'class' : 'private road', 'rgb' : [255,0,0]},
        6 : {'class' : 'Sidewalk', 'rgb' : [255,150,0]},
        7 : {'class' : 'Separated pedestrian path', 'rgb' : [150,255,0]},
    }

    rgb_to_labels = {
        '000000000' : 0,
        '150000000' : 1,
        '150150150' : 2,
        '150000150' : 3,
        '000150255' : 4,
        '255000000' : 5,
        '255150000' : 6,
        '150255000' : 7
    }
    
    labels, rgb_to_labels, translate_labels, num_classes = desired_classes_to_dict(labels,rgb_to_labels,desired_classes,background_index, num_classes)
        

    return get_train_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,translate_labels=translate_labels,
                                ignore_index=ignore_index,
                                num_classes=num_classes, img_dir=img_dir, mask_dir=mask_dir, 
                                train_folders=train_folders,val_folders=val_folders, 
                                find_criteria=find_criteria, img_to_mask=img_to_mask,
                                img_folder=img_folder,mask_folder=mask_folder)

def get_Vienna_test_datamodule(desired_classes = None, img_dir:str = "",test_folders="test",img_folder="img", 
                                find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    
    num_classes = 8
    ignore_index = None
    background_index = 0

    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'Road', 'rgb' : [150,0,0]},
        2 : {'class' : 'Tram', 'rgb' : [150,150,150]},
        3 : {'class' : 'Crosswalk', 'rgb' : [150,0,150]},
        4 : {'class' : 'Parking', 'rgb' : [0,150,255]},
        5 : {'class' : 'private road', 'rgb' : [255,0,0]},
        6 : {'class' : 'Sidewalk', 'rgb' : [255,150,0]},
        7 : {'class' : 'Separated pedestrian path', 'rgb' : [150,255,0]},
    }

    rgb_to_labels = {
        '000000000' : 0,
        '150000000' : 1,
        '150150150' : 2,
        '150000150' : 3,
        '000150255' : 4,
        '255000000' : 5,
        '255150000' : 6,
        '150255000' : 7
    }
    
    labels, rgb_to_labels, translate_labels, num_classes = desired_classes_to_dict(labels,rgb_to_labels,desired_classes,background_index, num_classes)
        
    return get_test_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,translate_labels=translate_labels,ignore_index=ignore_index,
                                num_classes=num_classes,img_dir=img_dir,test_folders=test_folders,
                                img_folder=img_folder, find_criteria=find_criteria)

def get_parkingPublicPrivate_train_datamodule(img_dir:str, mask_dir = None,  
                        train_folders = "", val_folders = "", img_folder = "img", mask_folder = "mask",
                        find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
                        img_to_mask:dict = {"image":"mask",".jpg":".png"}):
    
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        2 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
    }

    rgb_to_labels = {
        '255000000' : 1,
        '000255000' : 2,
        '000000000' : 0
    }

    return get_train_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                                num_classes=num_classes, img_dir=img_dir, mask_dir=mask_dir, 
                                train_folders=train_folders,val_folders=val_folders, 
                                find_criteria=find_criteria, img_to_mask=img_to_mask,
                                img_folder=img_folder,mask_folder=mask_folder)

def get_parkingPublicPrivate_test_datamodule(img_dir:str,test_folders="",img_folder="img",
                                find_criteria:dict = {"image":True,".jpg":True,".aux":False}):
    
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        2 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
    }

    rgb_to_labels = {
        '255000000' : 1,
        '000255000' : 2,
        '000000000' : 0
    }

    return get_test_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                                num_classes=num_classes,img_dir=img_dir,test_folders=test_folders,
                                img_folder=img_folder, find_criteria=find_criteria)




def get_Dubai_datamodule(data_dir: str):
    import os
    import ntpath
    import seg_utils as utils

    num_classes = 6
    ignore_index = 5
    labels = {
        0 : {'class' : 'building', 'rgb' : [60,16,152]},
        1 : {'class' : 'land', 'rgb' : [132,41,246]},
        2 : {'class' : 'road', 'rgb' : [110,193,228]},
        3 : {'class' : 'vegetation', 'rgb' : [254,221,58]},
        4 : {'class' : 'water', 'rgb' : [226,169,41]},
        5 : {'class' : 'unlabeled', 'rgb' : [155,155,155]},
    }

    rgb_to_labels = {
        '060016152' : 0,
        '132041246' : 1,
        '110193228' : 2,
        '254221058' : 3,
        '226169041' : 4,
        '155155155' : 5,
        '000000000' : 5,
    }

    tiles = [1,2,3,4,5,6,7,8]

    image_paths = []
    for i in tiles:
        f = data_dir + "/Tile " + str(i) + "/images" 
        for j in os.scandir(f):
            if j.is_file() and ".jpg" in str(j.name):
                image_paths.append(os.path.normpath(f + "/" + str(j.name)))
                
    mask_dict = {}
    for i in image_paths:
        s = i.replace("images","masks")
        s = s.replace(".jpg",".png")
        mask_dict[i] = os.path.normpath(s)

    return SegDataModule(num_classes=num_classes,train_paths=image_paths,train_mask_dict=mask_dict,
                            ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels)


def get_TUWien_train_datamodule(data_dir:str):
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        1 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
        2 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '000000000' : 2,
        '255000000' : 0,
        '000255000' : 1
    }

    train_folders = ["train"]
    val_folders = ["test/GT_1","test/GT_2","test/GT_3","test/GT_4"]
    find_criteria:dict = {"image":True,".jpg":True,".aux":False}, 
    img_to_mask:dict = {"image":"mask",".jpg":".png"}

    return get_train_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                            num_classes=num_classes,img_dir=data_dir,train_folders=train_folders,val_folders=val_folders,
                            img_folder="img",mask_folder="mask",
                            find_criteria=find_criteria,img_to_mask=img_to_mask)

def get_TUWien_test_datamodule(data_dir:str, ind = [1,2,3,4]):
    if type(ind) is int:
        ind = [ind]

    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        1 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
        2 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '000000000' : 2,
        '255000000' : 0,
        '000255000' : 1
    }

    find_criteria:dict = {"image":True,".jpg":True,".aux":False}

    test_folders = []
    for i in ind:
        test_folders.append(f"test/GT_{i}")

    return get_test_datamodule(labels=labels,rgb_to_labels=rgb_to_labels,ignore_index=ignore_index,
                                num_classes=num_classes,img_dir=data_dir,test_folders=test_folders,img_folder="img",
                                find_criteria=find_criteria)


def get_TUWienV0_datamodule(data_dir: str):
    import os
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'ulabeled', 'rgb' : [0,0,0]},
        1 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        2 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
    }

    rgb_to_labels = {
        '000000000' : 0,
        '255000000' : 1,
        '000255000' : 2
    }

    train_paths = []
    for i in os.scandir(data_dir + "/train/RGB_tiles"):
        if i.is_file() and ".tif" in str(i.name) and "RGB_" in str(i.name):
            train_paths.append(os.path.normpath(data_dir + "/train/RGB_tiles/" + str(i.name)))

    test_paths = []
    for i in os.scandir(data_dir + "/test/RGB_tiles"):
        if i.is_file() and ".tif" in str(i.name) and "RGB_" in str(i.name):
            test_paths.append(os.path.normpath(data_dir + "/test/RGB_tiles/" + str(i.name)))
    
    mask_dict = {}
    _train_paths = []
    for i in range(len(train_paths)):
        s1 = train_paths[i]
        s2 = s1.replace("RGB","GT")
        if os.path.isfile(s2):
            mask_dict[s1] = s2
            _train_paths.append(s1)
        else:
            print(f"Warning. Mask file {s2} not found for image {s1}")
    
    train_paths = _train_paths

    _test_paths = []
    for i in range(len(test_paths)):
        s1 = test_paths[i]
        s2 = s1.replace("RGB","GT")
        if os.path.isfile(s2):
            mask_dict[s1] = s2
            _test_paths.append(s1)
        else:
            print(f"Warning. Mask file {s2} not found for image {s1}")
    
    test_paths = _test_paths
    
    return SegDataModule(num_classes=num_classes,train_paths=train_paths,train_mask_dict=mask_dict, val_paths=test_paths,
                         ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels)

























def get_TUWienV2_datamodule(data_dir: str):
    import os
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        1 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
        2 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '000000000' : 2,
        '255000000' : 0,
        '000255000' : 1
    }

    train_paths = []
    for i in os.scandir(data_dir + "/train/img"):
        if i.is_file() and ".jpg" in str(i.name) and "image" in str(i.name) and ".aux" not in str(i.name):
            train_paths.append(os.path.normpath(data_dir + "/train/img/" + str(i.name)))


    test_paths = []
    for ii in [1,2,3,4]:
        for i in os.scandir(data_dir + f"/test/GT_{ii}/img"):
            if i.is_file() and ".jpg" in str(i.name) and "image" in str(i.name) and ".aux" not in str(i.name):
                test_paths.append(os.path.normpath(data_dir + f"/test/GT_{ii}/img/" + str(i.name)))
        
    mask_dict = {}
    _train_paths = []
    for i in range(len(train_paths)):
        s1 = train_paths[i]
        s2 = s1.replace("image","mask")
        s2 = s2.replace("/img/","/mask/")
        s2 = s2.replace(".jpg",".png")
        if os.path.isfile(s2):
            mask_dict[s1] = s2
            _train_paths.append(s1)
        else:
            print(f"Warning. Mask file {s2} not found for image {s1}")
    
    train_paths = _train_paths

    _test_paths = []
    for i in range(len(test_paths)):
        s1 = test_paths[i]
        s2 = s1.replace("image","mask")
        s2 = s2.replace("/img/","/mask/")
        s2 = s2.replace(".jpg",".png")
        if os.path.isfile(s2):
            mask_dict[s1] = s2
            _test_paths.append(s1)
        else:
            print(f"Warning. Mask file {s2} not found for image {s1}")
    
    test_paths = _test_paths
    
    return SegDataModule(num_classes=num_classes,train_paths=train_paths,train_mask_dict=mask_dict, val_paths=test_paths,
                         ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels)

def get_TUWienTestV2_datamodule(data_dir:str, idx:int):
    import os
    num_classes = 3
    ignore_index = None
    labels = {
        0 : {'class' : 'public_parking_space', 'rgb' : [255,0,0]},
        1 : {'class' : 'private_parking_space', 'rgb' : [0,255,0]},
        2 : {'class' : 'unlabeled', 'rgb' : [0,0,0]},
    }

    rgb_to_labels = {
        '000000000' : 2,
        '255000000' : 0,
        '000255000' : 1
    }

    test_paths = []
    for i in os.scandir(data_dir + f"/test/GT_{idx}/img"):
        if i.is_file() and ".jpg" in str(i.name) and "image" in str(i.name) and ".aux" not in str(i.name):
            test_paths.append(os.path.normpath(data_dir + f"/test/GT_{idx}/img/" + str(i.name)))

    mask_dict = {}
        
    _test_paths = []
    for i in range(len(test_paths)):
        s1 = test_paths[i]
        s2 = s1.replace("image","mask")
        s2 = s2.replace("/img/","/mask/")
        s2 = s2.replace(".jpg",".png")
        if os.path.isfile(s2):
            mask_dict[s1] = s2
            _test_paths.append(s1)
        else:
            print(f"Warning. Mask file {s2} not found for image {s1}")

    test_paths = _test_paths
    
    return SegDataModule(num_classes=num_classes,train_paths=None,train_mask_dict=None, val_paths=test_paths,
                         ignore_index=ignore_index,labels=labels,rgb_to_labels=rgb_to_labels)