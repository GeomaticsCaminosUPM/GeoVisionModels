def lables_to_image(img,labels,alpha=False):
    import numpy as np
    from PIL import Image

    if type(img) is Image:
        return img 
    
    s = img.shape
    l = max(np.array([*labels.keys()],dtype=np.uint8).max(),len(labels.keys()))
    if alpha:
        d = np.zeros((4,l), dtype=np.uint8)
    else:
        d = np.zeros((3,l), dtype=np.uint8)

    for i in labels.keys():
        d[0,i] = labels[i]['rgb'][0]
        d[1,i] = labels[i]['rgb'][1]
        d[2,i] = labels[i]['rgb'][2]
        if alpha:
            if type(alpha) is dict or type(alpha) is list:
                d[3,i] = alpha[labels[i]]
            else:
                d[3,i] = labels[i]['alpha'][3]
        
    if alpha:
        I = np.stack([d[0,img],d[1,img],d[2,img],d[3,img]])
        I = np.moveaxis(I,[0],[-1])
        return Image.fromarray(I,'RGBA')
    else:
        I = np.stack([d[0,img],d[1,img],d[2,img]])
        I = np.moveaxis(I,[0],[-1])
        return Image.fromarray(I,'RGB')

def tensor_to_image(img,mean=None,std=None):
    from PIL import Image
    import numpy as np
    import copy, torchvision

    #return torchvision.transforms.v2.ToPILImage()(img)

    if type(img) is Image:
        return img

    I = copy.copy(img.numpy())


    if type(std) != type(None) or type(mean) != type(None):
        ind = 0
        for j in range(len(I.shape)):
            if I.shape[j] == 3:
                ind = j
        k = np.ones(len(I.shape),dtype=np.uint8)
        k[ind] = 3
        if type(std) != type(None):
            if len(std) == 3:
                std = np.array(std).reshape(k)
            elif len(std) != 1:
                raise Exception(f"Shape of std is {std.shape} which is neither 3 nor 1")
            I *= std
        if type(mean) != type(None):
            if len(mean) == 3:
                mean = np.array(mean).reshape(k)
            elif len(mean) != 1:
                raise Exception(f"Shape of mean is {mean.shape} which is neither 3 nor 1")
            I += mean

    if (I < 0).any():
        _min = I.min()
        _max = I.max()
        I = (I - _min) / _max
        
    if (I < 1.5).all():
        I *= 255 
    
    if I.shape[-1] != 3:
        ind = 0
        for j in range(len(I.shape)):
            if I.shape[j] == 3:
                ind = j

        I = np.moveaxis(I,[ind],[-1])

    return Image.fromarray(np.array(I,dtype=np.uint8),'RGB')

def plot_seg_result(img,mask,ground_truth=None,labels=None,background_index=None, title="Segmentation img and mask"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import get_cmap
    import numpy as np
    import copy ############## arreglar si img o mask son tensor
    
    fig, ax = plt.subplots()

    extent = [0,img.size[0],0,img.size[1]]
    ax.imshow(img, extent=extent, aspect='auto', alpha=1,zorder=0)

    if ground_truth:
        gt_cmap = ListedColormap(np.array([[1.0,0.0,0.0,0.7],[0.0,0.0,0.0,0.0]]))
        _ground_truth = copy.copy(mask)
        _ground_truth[mask == ground_truth] = 1 
        _ground_truth[mask != ground_truth] = 0

        gt_norm = Normalize(vmin=0, vmax=255)

        ax.imshow(_ground_truth, extent=extent, aspect='auto', alpha=1, cmap=gt_cmap, norm=gt_norm, interpolation='none',zorder=1)

    if labels:
        colors = []
        ind = 0
        for i in labels.keys():
            c = list(np.array(labels[i]['rgb']) / np.array([255,255,255]))
            if len(c) == 3:
                c.append(1.0)
            
            if background_index == labels[i]:
                background_index = copy.copy(ind) 
        
            colors.append(c)
            ind += 1

        colors = np.array(colors)
        colors = np.clip(colors, 0, 1)
    else:
        # Define a custom colormap with transparency
        cmap = get_cmap('gist_rainbow')  # You can replace 'jet' with any other colormap of your choice
        colors = cmap(np.arange(cmap.N))
        # Increase the lightness of each color
        lightness_factor = 1.5  # Adjust the factor as needed (e.g., 1.0 for no change, values > 1 for lighter colors)
        colors = colors * lightness_factor
        colors = np.clip(colors, 0, 1)


    colors[:, -1] = 0.6  # Set alpha (transparency) to 0.5 for all colors

    if background_index:
        colors[background_index, -1] = 0.0  # Set alpha to 0.0 for value 255 (completely transparent)

    cmap = ListedColormap(colors)

    # Normalize the values to the range [0, 1]
    norm = Normalize(vmin=0, vmax=255)
    # Plot the NumPy array using Matplotlib
    ax.imshow(mask, extent=extent, aspect='auto', alpha=1, cmap=cmap, norm=norm, interpolation='none',zorder=1)


    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='datalim')  # Equal aspect ratio for x and y axes

    plt.title(title)
    plt.show() 
        
        
class SegmentationTransforms:
        
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target


    class Resize:
        def __init__(self, size,out_size=None):
            self.size = size
            if type(out_size) == type(None):
                out_size = size
            self.out_size = out_size

        def __call__(self, image, target):
            from torchvision.transforms import functional
            import torchvision.transforms.v2 as transforms
            image = functional.resize(image, self.size)
            target = functional.resize(target, self.out_size, interpolation=transforms.InterpolationMode.NEAREST)
            return image, target

    class RandomResize:
        def __init__(self, min_size, max_size=None):
            self.min_size = min_size
            if max_size is None:
                max_size = min_size
            self.max_size = max_size

        def __call__(self, image, target):
            import random
            from torchvision.transforms import functional
            import torchvision.transforms.v2 as transforms
            size = random.randint(self.min_size, self.max_size)
            image = functional.resize(image, size)
            target = functional.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
            return image, target

    class RandomCrop:
        def __init__(self, size):
            self.size = size

        def pad_if_smaller(img, size, fill=0):
            from torchvision.transforms import functional
            min_size = min(img.size)
            if min_size < size:
                ow, oh = img.size
                padh = size - oh if oh < size else 0
                padw = size - ow if ow < size else 0
                img = functional.pad(img, (0, 0, padw, padh), fill=fill)
            return img

        def __call__(self, image, target):
            from torchvision.transforms import functional
            image = self.pad_if_smaller(image, self.size)
            target = self.pad_if_smaller(target, self.size, fill=255)
            crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
            image = functional.crop(image, *crop_params)
            target = functional.crop(target, *crop_params)
            return image, target

    class RandomHorizontalFlip:
        def __init__(self, flip_prob=0.5):
            self.flip_prob = flip_prob

        def __call__(self, image, target):
            import random
            from torchvision.transforms import functional
            if random.random() < self.flip_prob:
                image = functional.hflip(image)
                target = functional.hflip(target)
            return image, target
        
    class RandomVerticalFlip:
        def __init__(self, flip_prob=0.5):
            self.flip_prob = flip_prob

        def __call__(self, image, target):
            import random
            from torchvision.transforms import functional
            if random.random() < self.flip_prob:
                image = functional.vflip(image)
                target = functional.vflip(target)
            return image, target
        
    class RandomRotation:
        def __init__(self, angle, rot_prob=0.2):
            self.rot_prob = rot_prob
            self.angle = angle

        def __call__(self, image, target):
            import random
            from torchvision.transforms import functional
            import torchvision.transforms.v2 as transforms
            if random.random() < self.rot_prob:
                image = functional.rotate(image,self.angle,interpolation=transforms.InterpolationMode.BILINEAR)
                target = functional.rotate(image,self.angle,interpolation=transforms.InterpolationMode.NEAREST)
            return image, target
        
    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, image, target):
            from torchvision.transforms import functional
            image = functional.normalize(image, mean=self.mean, std=self.std)
            return image, target
    
    class ColorJitter:
        def __init__(self, brightness=0.1, contrast=0.1, hue=0.1):
            import torchvision.transforms.v2 as transforms
            self.brightness = brightness
            self.contrast = contrast
            self.hue = hue
            self.trans = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, hue=self.hue)

        def __call__(self, image, target):
            image = self.trans(image)
            return image, target
        
    class toImage:
        def __init__(self):
            import torchvision.transforms.v2 as transforms
            self.trans = transforms.ToImage()

        def __call__(self, image, target):
            image = self.trans(image)
            target = self.trans(target)
            return image, target  
              
    class toTensor:
        def __init__(self):
            import torchvision.transforms.v2 as transforms
            import torch
            self.trans = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
            self.trans_b = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.uint8, scale=False)])
        def __call__(self, image, target):
            image = self.trans(image)
            target = self.trans_b(target)
            return image, target
    
    class toNumpy:
        def __init__(self):
            import numpy
            self.trans = numpy.array

        def __call__(self, image, target):
            image = self.trans(image)
            target = self.trans(target)
            return image, target

    class pretrainedTransforms:
        def __init__(self,encoder,pretrained):
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            import torchvision.transforms.v2 as transforms
            import torch
            self.trans = get_preprocessing_fn(encoder, pretrained=pretrained)
            self.toImage = transforms.ToImage()


        def __call__(self, image, target):
            import numpy
            image = numpy.array(image)
            image = self.toImage(self.trans(image))
            target = self.toImage(target)
            return image, target

aug_transforms_1 = SegmentationTransforms.Compose([
    SegmentationTransforms.RandomHorizontalFlip(flip_prob=0.5),
    SegmentationTransforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    ])

aug_transforms_2 = SegmentationTransforms.Compose([
    SegmentationTransforms.RandomHorizontalFlip(flip_prob=0.5),
    SegmentationTransforms.RandomVerticalFlip(flip_prob=0.5),
    SegmentationTransforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    ])

aug_transforms_3 = SegmentationTransforms.Compose([
    SegmentationTransforms.RandomHorizontalFlip(flip_prob=0.5),
    SegmentationTransforms.RandomVerticalFlip(flip_prob=0.5),
    SegmentationTransforms.RandomRotation(angle=20,rot_prob=0.2),
    SegmentationTransforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    ])


def translate_array(arr,translate,values=None):
    import numpy as np
    cond_list = []
    if type(translate) is dict:
        values = np.uint8(np.array(list(translate.values())))
        j = -1
        for i in translate.keys():
            j += 1
            if i != values[j]:
                cond_list.append(arr == i)

    elif type(values) != type(None):
        values = np.uint8(np.array(values))
        for i in range(len(translate)):
            if translate[i] != values[i]:
                cond_list.append(arr == translate[i])

    else:
        raise Exception(f"Type error in translate array. type(translate) is {type(translate)}.")
        
    return np.select(cond_list,values,arr)

def translate_mask(I,translate:dict,values=None):
    import numpy as np
    from PIL import Image
    im = np.uint8(I) 
    if len(im.shape) != 2:
        raise Exception(f"Mask shape is not valid {im.shape} but it should have 2 dimensions")
    
    mask = translate_array(im,translate,values)
    
    return Image.fromarray(np.uint8(mask),mode='L')

def clean_mask_by_kernel(in_img, kernel=(5,5)):
    import cv2
    import numpy as np

    # first do morphological opening to remove noise
    opening = cv2.morphologyEx(in_img, cv2.MORPH_OPEN, kernel=np.ones(kernel, np.uint8))
    # then do morphological closing to close small holes inside the detected areas
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=np.ones(kernel, np.uint8))
    return closing
    

def clean_mask_by_area(mask,areas,lengths,background_index = 0, open_first = True, footprint = "octagon", order = None):
    import skimage, copy
    import numpy as np

    mask_clean = copy.copy(mask)
    if type(areas) is dict:
        if areas.keys() != lengths.keys():
            raise Exception(f"Areas and lengths must have same keys but got. {areas.keys()} and {lengths.keys()}")
        
        translator = list(areas.keys())
        areas = list(areas.values())
        lengths = [lengths[i] for i in translator]
    else:
        translator = list(range(len(areas)))
    
    if order:
        inds = order
        if len(inds) != len(areas):
            raise Exception(f"order list does not contain all area values. {order}")
        
    else:
        inds = np.argsort(np.array(areas))

    bg_idx = mask == background_index
    inv_bg_idx = bg_idx == 0
    for i in range(len(inds)):
        if inds[i] == background_index:
            continue

        if footprint == "disk":
            _footprint = skimage.morphology.disk(int(round(lengths[inds[i]]/2.15)),decomposition='sequence')
        elif footprint == "octagon":
            _footprint = skimage.morphology.octagon(int(round(lengths[inds[i]]/2.15)),int(round(lengths[inds[i]]/3.75)),decomposition='sequence')
        elif footprint == "diamond":
            _footprint = skimage.morphology.diamond(int(round(lengths[inds[i]]/2.15)),decomposition='sequence')
        elif footprint == "star":
            _footprint = skimage.morphology.star(int(round(lengths[inds[i]]/3)))#,decomposition='sequence')
        elif footprint == "square":
            _footprint = [(np.ones((int(round(lengths[inds[i]])), 1)), 1), (np.ones((1, int(round(lengths[inds[i]])))), 1)]

        if i != 0:
            mask1 = mask_clean != translator[inds[0]]
            inv_mask1 = mask1 == 0
            m0 = int(round(min(areas[inds[i-1]]/2,areas[inds[i]]/2)))

            if m0 != int(round(min(areas))):
                res0 = skimage.morphology.remove_small_holes(mask1, m0)
                cond0 = res0 * inv_mask1 * inv_bg_idx
                mask_clean = cond0 * translator[inds[i]] + (cond0 == 0) * mask_clean
                
            res1 = skimage.morphology.remove_small_holes(mask1, int(round(min(areas))))
            cond1 = res1 * inv_mask1
            mask_clean = cond1 * translator[inds[i]] + (cond1 == 0) * mask_clean
    
        if open_first == True:
            mask2 = mask_clean == translator[inds[i]]
            res2 = skimage.morphology.binary_erosion(mask2,footprint=_footprint)
            res2 = skimage.morphology.remove_small_objects(res2,int(round(areas[inds[i]]/2)))
            res2 = skimage.morphology.binary_dilation(res2,footprint=_footprint)
            res2 = skimage.morphology.binary_dilation(res2,footprint=_footprint)
            res2 = skimage.morphology.remove_small_holes(res2, int(round(areas[inds[i]]/4)))
            res2 = skimage.morphology.binary_erosion(res2,footprint=_footprint)
            inv_res2 = res2 == 0
            res2_mask2 = res2 == mask2
            mask_clean = res2 * translator[inds[i]] + mask_clean * inv_res2 * res2_mask2 + translator[inds[0]] * (res2_mask2 == 0) * inv_res2
        else:
            mask2 = mask_clean == translator[inds[i]]
            res2 = skimage.morphology.binary_dilation(mask2,footprint=_footprint)
            res2 = skimage.morphology.remove_small_holes(res2, int(round(areas[inds[i]]/2)))
            res2 = skimage.morphology.binary_erosion(res2,footprint=_footprint)
            res2 = skimage.morphology.binary_erosion(res2,footprint=_footprint)
            res2 = skimage.morphology.remove_small_objects(res2, int(round(areas[inds[i]]/2)))
            res2 = skimage.morphology.binary_dilation(res2,footprint=_footprint)

        inv_res2 = res2 == 0
        res2_mask2 = res2 == mask2
        mask_clean = res2 * translator[inds[i]] + mask_clean * inv_res2 * res2_mask2 + translator[inds[0]] * (res2_mask2 == 0) * inv_res2

    return mask_clean


"""
def rgb_mask_to_indices(I,rgb_to_labels):
    #import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    im = np.uint8(I)
    if len(im.shape) == 2:
        return I
    else:
        i,j,_ = im.shape
    mask = np.ones((i,j), dtype="uint8") * 255
    for _i in range(i):
        for _j in range(j):
            s = ('000' + str(im[_i,_j,0]))[-3:] + ('000' + str(im[_i,_j,1]))[-3:] + ('000' + str(im[_i,_j,2]))[-3:]
            try:
                mask[_i,_j] = rgb_to_labels[s]
            except Exception as e:
                import warnings
                warnings.warn(f"Error while loading mask in indices [{_i,_j}]. Error message {str(e)}")
                return None

    return Image.fromarray(mask,mode='L')

def repair_mask_file(mask:str,rgb_to_labels=None,translate_labels=None,save_invalid=False):
    from PIL import Image 
    import copy

    I = Image.open(mask)

    if rgb_to_labels:
        _I = rgb_mask_to_indices(I,rgb_to_labels)
        if type(I) == type(None):
            raise Exception(f"Error while processing image {mask}")
        if I != _I:
            if save_invalid:
                I.save(save_invalid)
                print(f"original rgb mask file {mask} copied to {save_invalid}")
                save_invalid = False
            
            print(f"mask file {mask} saved with indices instead of rgb labels")
            _I.save(mask)
            I = copy.copy(_I)
    
    if translate_labels:
        _I = translate_labels(I, translate_labels)
        if I != _I:
            if save_invalid:
                I.save(save_invalid)
                print(f"mask file {mask} with original indicies copied to {save_invalid}")
                save_invalid = False
            
            print(f"mask file {mask} saved with new indices")
            _I.save(mask)


def repair_masks(mask_paths,rgb_to_labels=None,translate_labels=None,save_invalid=False):
    import os, ntpath

    if type(mask_paths) is dict:
        _mask_paths = []
        for i in mask_paths.keys():
            _mask_paths.appendI(mask_paths[i])

        mask_paths = _mask_paths

    for i in mask_paths:
        mask_path = os.path.normpath(mask_paths[i])

        if save_invalid:
            folder_list = []
            for j in mask_path.split("/"):
                for k in j.split("\\"):
                    folder_list.append(k)
            
            save_orig = folder_list[0]
            for j in [*folder_list[1:-2],"original_masks",*folder_list[-2:]]:
                save_orig += f"/{j}"
            
            save_orig = os.path.normpath(save_orig)
        else:
            save_orig = False

        head, tail = ntpath.split(save_orig)
        if not os.path.isdir(head):
            os.makedirs(head)
        
        repair_mask_file(i,rgb_to_labels=rgb_to_labels, translate_labels=translate_labels, save_invalid=save_orig)
"""
def get_mean_and_std(arg1,arg2=None):
    if type(arg1) is str and type(arg2) is type(None):
        model_name = arg1
        mean_and_std_dict = {
            'deeplabV3' : {'mean' : [0.485, 0.456, 0.406], 'std' :[0.229, 0.224, 0.225]},
        }
        print(f"Model needs images to have mean {mean_and_std_dict[model_name]['mean']} and std {mean_and_std_dict[model_name]['std']}")
        return mean_and_std_dict[model_name]['mean'], mean_and_std_dict[model_name]['std']
    elif type(arg1) is dict and type(arg2) is str: 
        model_name = arg2
        mean_and_std_dict = arg1
        print(f"Model needs images to have mean {mean_and_std_dict[model_name]['mean']} and std {mean_and_std_dict[model_name]['std']}")
        return mean_and_std_dict[model_name]['mean'], mean_and_std_dict[model_name]['std']
    elif type(arg1) is str and type(arg2) is str:
        from segmentation_models_pytorch.encoders import get_preprocessing_fn
        encoder = arg1
        pretrained = arg2
        trans = get_preprocessing_fn(encoder, pretrained=pretrained)
        print(f"Model needs images to have mean {trans.keywords['mean']} and std {trans.keywords['std']}")
        return trans.keywords['mean'], trans.keywords['std']
    elif type(arg1) is list and (type(arg2) is int or type(arg2) is tuple):
        import copy
        input_size = copy.copy(arg2)
        image_paths = copy.copy(arg1)
        if type(arg2) is int:
            input_size = (input_size,input_size)
        mean, std = compute_img_mean_std(image_paths=image_paths,input_size=input_size)
        print(f"Dataset mean {mean} and std {std}")
        return mean, std
    else:
        raise Exception(f"""Type of arg1 is {type(arg1)} but should be str (model name) or str (encoder name) or dict (model_name : mean, std) or list of dataset img paths.\n
                        Type of arg1 is {type(arg2)} but should be None or str (pretrained weigths) or str (model_name) or tuple or int (image size)""")

def compute_img_mean_std(image_paths,input_size):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    import cv2
    from tqdm import tqdm
    import numpy as np
    import warnings

    img_h, img_w = input_size
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths)), desc="Computing mean and std of imgaes: "):
        try:
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (img_h, img_w))
        except:
            warnings.warn(f"Warning. img {image_paths[i]} is corrupt")
            raise Exception(f"Warning. img {image_paths[i]} is corrupt")
            continue

        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    return means,stdevs

def get_countmatrix(dataloader,num_classes:int=0):
    import numpy as np
    if type(dataloader) == type(None):
        raise Exception("dataloader not set") 
    
    try:
        return dataloader.countmatrix
    except:
        None 

    try:
        num_classes = dataloader.num_classes
    except:
        if num_classes < 2:
            raise Exception(f"num_classes must be >= 2 but got {num_classes}")
        
    if num_classes < 2:
        raise Exception(f"num_classes must be >= 2 but got {num_classes}. Maybe your datamodule does not have a num_classes field.")
    

    try:
        dataloader = dataloader.train_dataloader() 
        print("Warning. get_countmatrix() got a datamodule instead of a dataloader. Selecting the train_dataloader.")
    except:
        None
    
    bincounts = np.zeros((len(dataloader) * dataloader.batch_size,num_classes),dtype=int)
    i=0
    for _,y in iter(dataloader):
        for j in range(y.shape[0]):
            _y = y.detach().numpy().astype('uint8')
            bincounts[i,:] = np.matrix(np.bincount(np.squeeze(np.asarray(_y[j].flatten())),minlength=num_classes))
            i += 1

    return np.matrix(bincounts)
    
def get_class_weights(dataloader=None,num_classes:int=0,method:str='ratio',balanced:bool = False,normalize:bool = True, countmatrix=None):
    import numpy as np

    if type(countmatrix) == type(None):
        countmatrix = get_countmatrix(dataloader,num_classes)

    num_classes = countmatrix.shape[1]

    if method == 'pixel':
        bincounts = np.sum(countmatrix,axis=0)
        if balanced:
            n_samples = np.sum(np.multiply(np.repeat(np.sum(countmatrix,axis=1),countmatrix.shape[1],axis=1),(countmatrix > 0)),axis=0)
        else:
            n_samples = np.sum(countmatrix)

    elif method == 'area':
        bincounts = np.sum(np.divide(countmatrix, np.repeat(np.sum(countmatrix,axis=1),countmatrix.shape[1],axis=1)),axis=0)
        if balanced:
            n_samples = np.sum(countmatrix > 0,axis=0)
        else:
            n_samples = countmatrix.shape[0]
            
    else:
        raise Exception(f"method {method} not implemented")

    #w = (1 - np.squeeze(np.asarray(np.divide(bincounts, n_samples)))) * num_classes
    w = np.squeeze(np.asarray(np.divide(n_samples, num_classes * bincounts).flatten()))

    if normalize:
        w = (w / np.sum(w)) * num_classes
    
    return list(w)

def get_balanced_inds(countmatrix,n):
    import numpy as np

    median = np.squeeze(np.asarray(np.median(countmatrix,axis=0)))
    dot_prods = np.squeeze(np.asarray(np.dot(countmatrix,median)))
    order = np.argsort(dot_prods)
    sum_dot_prods = np.sum(dot_prods)
    repeat_array = np.round(((sum_dot_prods - dot_prods) / np.sum(sum_dot_prods - dot_prods)) * n).astype(int)
    if np.sum(repeat_array) < n:
        n_sum = n - np.sum(repeat_array)
        n_zero = np.sum(repeat_array == 0)
        if n_zero <= n_sum:
            repeat_array[len(repeat_array)-1:len(repeat_array)-1-n_zero:-1] = 1 
            n_sum -= n_zero
            repeat_array_b = np.ceil(((sum_dot_prods - dot_prods) / np.sum(sum_dot_prods - dot_prods)) * n_sum).astype(int)
            i = 0
            while n_sum > 0: 
                ra = repeat_array[i]
                rb = repeat_array_b[i]
                if rb - ra >= n_sum:
                    r = rb-ra 
                else:
                    r = n_sum
                
                repeat_array[i] += r 
                n_sum -= r
                i += 1
                if i > len(repeat_array)-1:
                    i = 0
        else:
            repeat_array[len(repeat_array)-1:len(repeat_array)-1-n_sum:-1]  = 1

    inds = np.repeat(np.arange(len(repeat_array)),repeat_array)
    return inds[len(inds)-n:len(inds)]

def delete_empty_masks(dataloader=None,num_classes:int=0,countmatrix=None, min_num_classes=0, min_area=0):
    import numpy as np
    import copy

    if min_num_classes < 2:
        if type(dataloader) != type(None):
            l = len(dataloader)
        elif type(countmatrix) != type(None):
            l = countmatrix.shape[0]
        else:
            return None, countmatrix

        return np.arange(l), countmatrix
    
    if type(countmatrix) == type(None):
        _countmatrix = get_countmatrix(dataloader,num_classes)
    else:
        _countmatrix = copy.deepcopy(countmatrix)

    if min_area > 0:
        _countmatrix = np.divide(_countmatrix, np.repeat(np.sum(_countmatrix,axis=1),_countmatrix.shape[1],axis=1))
        _countmatrix[_countmatrix < min_area] = 0

    num_classes = _countmatrix.shape[1]

    inds = np.arange(_countmatrix.shape[0])
    inds = inds[np.squeeze(np.array(np.sum(_countmatrix > 0, axis = 1).flatten())) >= min_num_classes]
    return inds, countmatrix[inds,:]

def patchify(_img,patch_size,step=None):
    import patchify, numpy

    img = numpy.array(_img)
    i = img.shape[-2]
    j = img.shape[-1]
    if i/patch_size[0] < 2 and j/patch_size[1] < 2:
        return img.reshape((1,*img.shape))
    
    if type(step) == type(None):
        step = patch_size
    all_img_patches = []  
    patches_img = patchify.patchify(img, patch_size, step=step)  #Step=256 for 256 patches means no overlap
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            if len(img.shape) == 2:
                single_patch_img = patches_img[i,j,:,:]
            else:
                single_patch_img = patches_img[i,j,:,:,:]
            all_img_patches.append(single_patch_img)
    return numpy.array(all_img_patches)


def set_parameter_requires_grad(model, requires_grad):
    """
    Congela todas las capas si feature_extracting == True
    """
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    elif requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False

def count_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        total += nn
        if p.requires_grad:
            trainable += nn 
    print(f"Model trainable params {trainable}, total params {total}")
    return trainable, total
