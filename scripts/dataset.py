from .import import *

class MultiClassCelebA(Dataset):
    
    def __init__(self, dataframe, folder_dir, transform = None):
        
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = dataframe.labels.values.tolist()
        
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.labels[index][0]
        sample = {'image': image, 'label': label.astype(float)}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': label.astype(float)}
        
        return sample


def get_label_df(path):
    labels_df = pd.read_csv(path)
    _dict = {}
    for i in range(1, len(label_df)):
        _dict[label_df['202599'][i].split()[0]] = [x for x in labels_df['202599'][i].split()[1:]]
    label_df = pd.DataFrame(_dict).T
    label_df.columns = (labels_df['202599'][0]).split()
    label_df.replace(['-1'], ['0'], inplace = True)
    return label_df



def get_dataframe(dir_path, label_df, df_for = 'train'):
    files = glob(dir_path+'/*jpg')
    n = len(files)
    n_imgs = round(files * 0.7)
    n_v = round(files*0.3)
    shuffle = np.random.permutation(n)
    _dict = {}
    _f_names = []
    if df_for is 'train':
        os.mkdir('train')
        for i in tqdm(shuffle(n_imgs:n)):
            file_name = files[i].split('/')[-1]
            labels = np.array(label_df[label_df.index == file_name])
            _dict[file_name] = labels
            _file_names.append(file_name)
            os.rename(files[i], os.path.join('train', file_name))
        train_df = pd.DataFrame(_dict.values())
        train_df.index = _file_names
        train_df.columns = ['labels']
        return train_df
    if df_for is 'valid':
        os.mkdir('valid')
        for i in tqdm(shuffle(:n_v)):
            file_name = files[i].split('/')[-1]
            labels = np.array(label_df[label_df.index == file_name])
            _dict[file_name] = labels
            _file_names.append(file_name)
            os.rename(files[i], os.path.join('valid', file_name))
        valid_df = pd.DataFrame(_dict.values())
        valid_df.index = _file_names
        valid_df.columns = ['labels']
        return valid_df
    else:
        return "Please provide 'train' or 'valid'"



