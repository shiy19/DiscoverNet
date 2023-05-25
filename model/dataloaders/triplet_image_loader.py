from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, conditions, split, n_triplets, aug=False, loader=default_image_loader):
        """ triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        filenames_filename = 'filenames.json'
        self.root = '/home/******'
        self.base_path = 'ut-zap50k-images' 
        self.split = split 
        self.filenamelist = []

        filenames = {'train': ['class_tripletlist_train.txt', 'closure_tripletlist_train.txt', 
                        'gender_tripletlist_train.txt', 'heel_tripletlist_train.txt'],
                    'val': ['class_tripletlist_val.txt', 'closure_tripletlist_val.txt', 
                        'gender_tripletlist_val.txt', 'heel_tripletlist_val.txt'],
                    'test': ['class_tripletlist_test.txt', 'closure_tripletlist_test.txt', 
                        'gender_tripletlist_test.txt', 'heel_tripletlist_test.txt'],
                    'train_half': ['class_tripletlist_train_half_reverse.txt', 'closure_tripletlist_train_half_reverse.txt', 
                        'gender_tripletlist_train_half_reverse.txt', 'heel_tripletlist_train_half_reverse.txt'],
                    'val_half': ['class_tripletlist_val_half_reverse.txt', 'closure_tripletlist_val_half_reverse.txt', 
                        'gender_tripletlist_val_half_reverse.txt', 'heel_tripletlist_val_half_reverse.txt'],
                    'test_half': ['class_tripletlist_test_half_reverse.txt', 'closure_tripletlist_test_half_reverse.txt', 
                        'gender_tripletlist_test_half_reverse.txt', 'heel_tripletlist_test_half_reverse.txt'],
                    'train_unsup': 'unsup_train_20w.txt'
                        }

        for line in open(os.path.join(self.root, filenames_filename)):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        if split == 'train':
            fnames = filenames['train']
        elif split == 'val':
            fnames = filenames['val']
        elif split == 'test':
            fnames = filenames['test']
        elif split == 'train_half':
            fnames = filenames['train_half']
        elif split == 'val_half':
            fnames = filenames['val_half']
        elif split == 'test_half':
            fnames = filenames['test_half']
        elif split == 'train_unsup':
            fnames = filenames['train_unsup']
        else:
            raise ValueError('Split error!')
        if split == 'train' or split == 'val' or split == 'test':
            for condition in list(range(conditions)):
                for line in open(os.path.join(self.root, 'tripletlists', fnames[condition])):
                    triplets.append((line.split()[0], line.split()[1], line.split()[2], condition)) # anchor, far, close
        elif split == 'train_id':
            for condition in list(range(conditions)):
                for i,line in enumerate(open(os.path.join(self.root, 'tripletlists', fnames[condition]))):
                    triplets.append((line.split()[0], line.split()[1], line.split()[2], condition, i)) # anchor, far, close, index        
        elif split == 'train_unsup':
            for i,line in enumerate(open(os.path.join(self.root, 'tripletlists', fnames))):
                triplets.append((line.split()[0], line.split()[1], line.split()[2])) 
        else:
            for condition in list(range(conditions)):
                for line in open(os.path.join(self.root, 'tripletlists', fnames[condition])):
                    triplets.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3], condition))                
        
        np.random.shuffle(triplets)
        self.triplets = triplets[:n_triplets]
        self.loader = loader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if (split == 'train' or split == 'train_half' or split == 'train_unsup') and aug:
            self.transform=transforms.Compose([
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform=transforms.Compose([
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                normalize,
            ])        

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            path1, path2, path3, c = self.triplets[index]
            if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
                img1 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]))
                img2 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]))
                img3 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]))
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, c
        else:
            path1, path2, path3, label, c = self.triplets[index]
            if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
                img1 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]))
                img2 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]))
                img3 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]))
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return img1, img2, img3, label, c 

    def __len__(self):
        return len(self.triplets)

class TripletImageLoader_celeba(torch.utils.data.Dataset):
    def __init__(self, args, split, n_triplets, aug=False, loader=default_image_loader):
        """ triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.root = '/home/******'
        self.base_path = 'img_align_celeba'
        self.conditions = args.ncondition  
        self.split = split
        if self.conditions==8:
            filenames_celeba = {'train': ['0_train.txt', '2_train.txt', '3_train.txt', '15_train.txt', '20_train.txt', '31_train.txt', '36_train.txt', '39_train.txt'],
                    'val': ['0_val.txt', '2_val.txt', '3_val.txt', '15_val.txt', '20_val.txt', '31_val.txt', '36_val.txt', '39_val.txt'],
                    'test': ['0_test.txt', '2_test.txt', '3_test.txt', '15_test.txt', '20_test.txt', '31_test.txt', '36_test.txt', '39_test.txt'],
                    'train_half': ['0_train_half_reverse.txt', '2_train_half_reverse.txt', '3_train_half_reverse.txt', '15_train_half_reverse.txt', '20_train_half_reverse.txt', '31_train_half_reverse.txt', '36_train_half_reverse.txt', '39_train_half_reverse.txt'],
                    'val_half': ['0_val_half_reverse.txt', '2_val_half_reverse.txt', '3_val_half_reverse.txt', '15_val_half_reverse.txt', '20_val_half_reverse.txt', '31_val_half_reverse.txt', '36_val_half_reverse.txt', '39_val_half_reverse.txt'],
                    'test_half': ['0_test_half_reverse.txt', '2_test_half_reverse.txt', '3_test_half_reverse.txt', '15_test_half_reverse.txt', '20_test_half_reverse.txt', '31_test_half_reverse.txt', '36_test_half_reverse.txt', '39_test_half_reverse.txt'],
                    }
        elif self.conditions==40:
            filenames_celeba = {'train': ['0_train.txt', '1_train.txt', '2_train.txt', '3_train.txt', '4_train.txt', '5_train.txt', '6_train.txt', '7_train.txt', '8_train.txt', '9_train.txt',
                                '10_train.txt', '11_train.txt', '12_train.txt', '13_train.txt', '14_train.txt', '15_train.txt', '16_train.txt', '17_train.txt', '18_train.txt', '19_train.txt',
                                '20_train.txt', '21_train.txt', '22_train.txt', '23_train.txt', '24_train.txt', '25_train.txt', '26_train.txt', '27_train.txt', '28_train.txt', '29_train.txt',
                                '30_train.txt', '31_train.txt', '32_train.txt', '33_train.txt', '34_train.txt', '35_train.txt', '36_train.txt', '37_train.txt', '38_train.txt', '39_train.txt'],
                    'val':  ['0_val.txt', '1_val.txt', '2_val.txt', '3_val.txt', '4_val.txt', '5_val.txt', '6_val.txt', '7_val.txt', '8_val.txt', '9_val.txt',
                                '10_val.txt', '11_val.txt', '12_val.txt', '13_val.txt', '14_val.txt', '15_val.txt', '16_val.txt', '17_val.txt', '18_val.txt', '19_val.txt',
                                '20_val.txt', '21_val.txt', '22_val.txt', '23_val.txt', '24_val.txt', '25_val.txt', '26_val.txt', '27_val.txt', '28_val.txt', '29_val.txt',
                                '30_val.txt', '31_val.txt', '32_val.txt', '33_val.txt', '34_val.txt', '35_val.txt', '36_val.txt', '37_val.txt', '38_val.txt', '39_val.txt'],
                    'test':  ['0_test.txt', '1_test.txt', '2_test.txt', '3_test.txt', '4_test.txt', '5_test.txt', '6_test.txt', '7_test.txt', '8_test.txt', '9_test.txt',
                                '10_test.txt', '11_test.txt', '12_test.txt', '13_test.txt', '14_test.txt', '15_test.txt', '16_test.txt', '17_test.txt', '18_test.txt', '19_test.txt',
                                '20_test.txt', '21_test.txt', '22_test.txt', '23_test.txt', '24_test.txt', '25_test.txt', '26_test.txt', '27_test.txt', '28_test.txt', '29_test.txt',
                                '30_test.txt', '31_test.txt', '32_test.txt', '33_test.txt', '34_test.txt', '35_test.txt', '36_test.txt', '37_test.txt', '38_test.txt', '39_test.txt'],
                    'train_half':  ['0_train_half_reverse.txt', '1_train_half_reverse.txt', '2_train_half_reverse.txt', '3_train_half_reverse.txt', '4_train_half_reverse.txt', '5_train_half_reverse.txt', '6_train_half_reverse.txt', '7_train_half_reverse.txt', '8_train_half_reverse.txt', '9_train_half_reverse.txt',
                                '10_train_half_reverse.txt', '11_train_half_reverse.txt', '12_train_half_reverse.txt', '13_train_half_reverse.txt', '14_train_half_reverse.txt', '15_train_half_reverse.txt', '16_train_half_reverse.txt', '17_train_half_reverse.txt', '18_train_half_reverse.txt', '19_train_half_reverse.txt',
                                '20_train_half_reverse.txt', '21_train_half_reverse.txt', '22_train_half_reverse.txt', '23_train_half_reverse.txt', '24_train_half_reverse.txt', '25_train_half_reverse.txt', '26_train_half_reverse.txt', '27_train_half_reverse.txt', '28_train_half_reverse.txt', '29_train_half_reverse.txt',
                                '30_train_half_reverse.txt', '31_train_half_reverse.txt', '32_train_half_reverse.txt', '33_train_half_reverse.txt', '34_train_half_reverse.txt', '35_train_half_reverse.txt', '36_train_half_reverse.txt', '37_train_half_reverse.txt', '38_train_half_reverse.txt', '39_train_half_reverse.txt'],            
                    'val_half':  ['0_val_half_reverse.txt', '1_val_half_reverse.txt', '2_val_half_reverse.txt', '3_val_half_reverse.txt', '4_val_half_reverse.txt', '5_val_half_reverse.txt', '6_val_half_reverse.txt', '7_val_half_reverse.txt', '8_val_half_reverse.txt', '9_val_half_reverse.txt',
                                '10_val_half_reverse.txt', '11_val_half_reverse.txt', '12_val_half_reverse.txt', '13_val_half_reverse.txt', '14_val_half_reverse.txt', '15_val_half_reverse.txt', '16_val_half_reverse.txt', '17_val_half_reverse.txt', '18_val_half_reverse.txt', '19_val_half_reverse.txt',
                                '20_val_half_reverse.txt', '21_val_half_reverse.txt', '22_val_half_reverse.txt', '23_val_half_reverse.txt', '24_val_half_reverse.txt', '25_val_half_reverse.txt', '26_val_half_reverse.txt', '27_val_half_reverse.txt', '28_val_half_reverse.txt', '29_val_half_reverse.txt',
                                '30_val_half_reverse.txt', '31_val_half_reverse.txt', '32_val_half_reverse.txt', '33_val_half_reverse.txt', '34_val_half_reverse.txt', '35_val_half_reverse.txt', '36_val_half_reverse.txt', '37_val_half_reverse.txt', '38_val_half_reverse.txt', '39_val_half_reverse.txt'],
                    'test_half':  ['0_test_half_reverse.txt', '1_test_half_reverse.txt', '2_test_half_reverse.txt', '3_test_half_reverse.txt', '4_test_half_reverse.txt', '5_test_half_reverse.txt', '6_test_half_reverse.txt', '7_test_half_reverse.txt', '8_test_half_reverse.txt', '9_test_half_reverse.txt',
                                '10_test_half_reverse.txt', '11_test_half_reverse.txt', '12_test_half_reverse.txt', '13_test_half_reverse.txt', '14_test_half_reverse.txt', '15_test_half_reverse.txt', '16_test_half_reverse.txt', '17_test_half_reverse.txt', '18_test_half_reverse.txt', '19_test_half_reverse.txt',
                                '20_test_half_reverse.txt', '21_test_half_reverse.txt', '22_test_half_reverse.txt', '23_test_half_reverse.txt', '24_test_half_reverse.txt', '25_test_half_reverse.txt', '26_test_half_reverse.txt', '27_test_half_reverse.txt', '28_test_half_reverse.txt', '29_test_half_reverse.txt',
                                '30_test_half_reverse.txt', '31_test_half_reverse.txt', '32_test_half_reverse.txt', '33_test_half_reverse.txt', '34_test_half_reverse.txt', '35_test_half_reverse.txt', '36_test_half_reverse.txt', '37_test_half_reverse.txt', '38_test_half_reverse.txt', '39_test_half_reverse.txt']}
        elif self.conditions==5:
            filenames_celeba = {'train': ['1_train.txt', '2_train.txt', '3_train.txt', '4_train.txt', '5_train.txt'],
                        'val': ['1_valid.txt', '2_valid.txt', '3_valid.txt', '4_valid.txt', '5_valid.txt'],
                        'test': ['1_test.txt', '2_test.txt', '3_test.txt', '4_test.txt', '5_test.txt'],
                         'train_half': ['1_train_half_reverse.txt', '2_train_half_reverse.txt', '3_train_half_reverse.txt', '4_train_half_reverse.txt', '5_train_half_reverse.txt'],
                        'val_half': ['1_valid_half_reverse.txt', '2_valid_half_reverse.txt', '3_valid_half_reverse.txt', '4_valid_half_reverse.txt', '5_valid_half_reverse.txt'],
                        'test_half': ['1_test_half_reverse.txt', '2_test_half_reverse.txt', '3_test_half_reverse.txt', '4_test_half_reverse.txt', '5_test_half_reverse.txt']}
        else:
            print("No such ncondition!")
        triplets = []
        if split == 'train':
            fnames = filenames_celeba['train']
        elif split == 'val':
            fnames = filenames_celeba['val']
        elif split == 'test':
            fnames = filenames_celeba['test']
        elif split == 'train_half':
            fnames = filenames_celeba['train_half']
        elif split == 'val_half':
            fnames = filenames_celeba['val_half']
        elif split == 'test_half':
            fnames = filenames_celeba['test_half']
        else:
            raise ValueError('Split error!')

        if self.conditions == 5:
            pathname = 'all_triplets_combineattr'
        else:
            pathname = 'all_triplets_split'

        if split == 'train' or split == 'val' or split == 'test':    
            for condition in list(range(self.conditions)):
                for line in open(os.path.join(self.root, pathname, fnames[condition])):
                    triplets.append((line.split()[0], line.split()[1], line.split()[2], condition)) # anchor, far, close 
        else:
            for condition in list(range(self.conditions)):
                for line in open(os.path.join(self.root, pathname, fnames[condition])):
                    triplets.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3], condition))   
 
        np.random.shuffle(triplets)
        self.triplets = triplets[:n_triplets]
        self.loader = loader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if (split == 'train' or split == 'train_half') and aug:
            self.transform=transforms.Compose([
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform=transforms.Compose([
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                normalize,
            ])        

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            path1, path2, path3, c = self.triplets[index]
            img1 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path1))
            img2 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path2))
            img3 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path3))
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            return img1, img2, img3, c
        elif self.split == 'train_unsup':
            path1, path2, path3 = self.triplets[index]
            img1 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path1))
            img2 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path2))
            img3 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path3))
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            return img1, img2, img3
        else:
            path1, path2, path3, label, c = self.triplets[index]
            img1 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path1))
            img2 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path2))
            img3 = self.loader(os.path.join(self.root, self.base_path, '{}.jpg').format(path3))
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            return img1, img2, img3, label, c            

    def __len__(self):
        return len(self.triplets)