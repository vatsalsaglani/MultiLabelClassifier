from .imports import *

from model import *

labels = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair',
 'Blond_Hair', 'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup',
 'High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
 'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat',
 'Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']
 
model_path = '#'

def load_model(path, div='cpu'):
    model = MultiClassifier()
    if div is 'cpu':
        model = torch.load(path, map_location='cpu')
        return model
    else:
        model = torch.load(path)
        return model



model = load_model(model_path)


def get_tensor(img):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return tfms(Image.open(img)).unsqueeze(0)

def predict(img, label_lst, model):
    tnsr = get_tensor(img)
    op = model(tnsr)
    op_b = torch.round(op)
    
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()
    
    preds = np.where(op_b_np == 1)[1]
    
    print(preds)
    
    label = []
    for i in preds:
        label.append(label_lst[i])
        
    return label, op

# predict(img_path, labels, model)