from .import import *
from model import *
from dataset import *
from acc import *
from model_fit import *


img_folder = '#'
label_df_path = '#'

label_df = get_label_df(label_df_path)

train_df = get_dataframe(img_folder, label_df, 'train')
valid_df = get_dataframe(img_folder, label_df, 'valid')

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dl = MultiClassCelebA(train_df, 'train/', transforms = tfms)
valid_dl = MultiClassCelebA(valid_df, 'valid/', transforms = tfms)

is_cuda = check_cuda()

model = MultiClassifier()
if is_cuda:
    model.cuda()

train_loader = DataLoader(train_dl, shuffle=True, batch_size=64, num_workers=3)
valid_loader = DataLoader(valid_dl, shuffle=True, batch_size=64, num_workers=3)


criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)


trn_losses = []; trn_acc = []
val_losses = []; val_acc = []
for i in tqdm(range(1, 20)):
    trn_l, trn_a = fit_model(i, model, train_dataloader)
    val_l, val_a = fit_model(i, model, valid_dataloader, phase = 'validation')
    trn_losses.append(trn_l); trn_acc.append(trn_a)
    val_losses.append(val_l); val_acc.append(val_a)
