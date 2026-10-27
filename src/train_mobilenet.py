# Training template for MobileNetV2 transfer learning.
# For a reproducible final comparison, use the exact split/transforms from your Colab notebook.
import argparse, os, copy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

def main():
    p=argparse.ArgumentParser(); p.add_argument("--train-dir",required=True); p.add_argument("--output-dir",default="models"); p.add_argument("--epochs",type=int,default=5); a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True); torch.manual_seed(42)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ColorJitter(.15,.15,.15),transforms.ToTensor(),norm])
    eval_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])
    ds=datasets.ImageFolder(a.train_dir,transform=train_tf); val_ds=datasets.ImageFolder(a.train_dir,transform=eval_tf)
    nv=max(1,round(.15*len(ds))); nt=len(ds)-nv
    tr,va=random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(42)); va=Subset(val_ds,va.indices)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl=DataLoader(tr,batch_size=32,shuffle=True,num_workers=0); vl=DataLoader(va,batch_size=32,shuffle=False,num_workers=0)
    model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for q in model.features.parameters(): q.requires_grad=False
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(ds.classes)); model.to(device)
    opt=torch.optim.Adam(model.classifier.parameters(),lr=.001); loss_fn=nn.CrossEntropyLoss()
    best,best_state=0,copy.deepcopy(model.state_dict())
    for ep in range(a.epochs):
        model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        model.eval(); correct=total=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device); pred=model(x).argmax(1); correct+=(pred==y).sum().item(); total+=len(y)
        acc=correct/max(total,1); print(f"Epoch {ep+1}: val_accuracy={acc:.4f}")
        if acc>best: best=acc; best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save({"state_dict":model.state_dict(),"classes":ds.classes,"architecture":"mobilenet_v2","image_size":224},os.path.join(a.output_dir,"mobilenetv2_best.pth"))
    print("Best validation accuracy:",best)
if __name__=="__main__": main()
