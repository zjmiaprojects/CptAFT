from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from tensorboardX import SummaryWriter
import os
from PIL import Image
from clip.CptAFT import CptAFT
# from clip.CptAFT import classification_AE as classification
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
class Dataset_load(Dataset):
    def __init__(self,meta_root,is_train,preprocess,tokenizer,context_length):
        # 1.根目录(根据自己的情况更改)
        self.meta_root = meta_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        # self.train_set_file = os.path.join(meta_root,'train_concept.txt')
        self.train_set_file = os.path.join(meta_root, 'train_label_concept.txt')
        # self.train_set_file = os.path.join(meta_root, 'train_single_concept.txt')
        self.test_set_file = os.path.join(meta_root,'val_label_concept.txt')
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_text = []
        self.sam_labels = []
        self.sam_concept_lab = []
        # 5.1 训练还是测试数据集
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
            with open(self.read_file, 'r') as f:
                for line in f:
                    img_path = line.strip().split('\t')[0]
                    label_text = line.strip().split('\t')[1]
                    # label = line.strip().split('\t')[2]
                    # text = label_text
                    pigment_network = line.strip().split('\t')[2]
                    streaks = line.strip().split('\t')[3]
                    regression_structures = line.strip().split('\t')[4]
                    dots_and_globules = line.strip().split('\t')[5]
                    blue_whitish_veil = line.strip().split('\t')[6]
                    label = line.strip().split('\t')[7]
                    concept_lab_text = line.strip().split('\t')[8]
                    num_list = [int(num) for num in concept_lab_text[1:-1].split(", ")]
                    concept_lab = torch.tensor(num_list)
                    # "photo of " + label_text+", because "+
                    text = "pigment_network is " + pigment_network + ", streaks is " + streaks \
                           + ", regression_structures is " + regression_structures + ", dots_and_globules is " + dots_and_globules + ", blue_whitish_veil is " + blue_whitish_veil
                    self.samples.append(img_path)
                    self.sam_text.append(text)
                    self.sam_labels.append(label)
                    self.sam_concept_lab.append(concept_lab)
            # 转换为token
            self.tokens = tokenizer(self.sam_text, context_length=context_length)
            # self.tokens = self.sam_text
        else:
            self.read_file = self.test_set_file
            with open(self.read_file, 'r') as f:
                for line in f:
                    img_path = line.strip().split('\t')[0]
                    label_text = line.strip().split('\t')[1]
                    label = line.strip().split('\t')[7]
                    # "photo of " + label_text+", because "+
                    text = label_text
                    self.samples.append(img_path)
                    self.sam_text.append(text)
                    self.sam_labels.append(label)
            # 转换为token
            self.tokens = tokenizer(self.sam_text, context_length=context_length)
            # self.tokens = self.sam_text
		# 5.2 获得所有的样本(根据自己的情况更改)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train==True:
            img_path = self.samples[idx]
            token = self.tokens[idx]
            label=int(self.sam_labels[idx])
            concept_lab=self.sam_concept_lab[idx]
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            # 对图像进行转换
            image = self.img_process(image)
            # image = img_path
            return image,token,label,concept_lab
        else:
            img_path = self.samples[idx]
            token = self.tokens[idx]
            label = int(self.sam_labels[idx])
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            # 对图像进行转换
            image = self.img_process(image)
            # image = img_path
            return image, token, label

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    net_biomedCLIP, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='model_pt/')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # checkpoint = {
    #     'network': net_biomedCLIP.state_dict()}
    # torch.save(checkpoint, "model_pt/biomedCLIP.pt")
    context_length = 128
    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    prediction_criterion = nn.CrossEntropyLoss()
    h_reconst_criterion = F.mse_loss
    concept_reconst_criterion = nn.BCELoss()
    # 加载数据集
    train_dataset = Dataset_load(meta_root='datasets/Derm7pt/data/', is_train=True,preprocess=preprocess,tokenizer=tokenizer,context_length=128)
    # train_dataset = Dataset_load(meta_root='datasets/PH2Dataset/data/', is_train=True,
    #                            preprocess=preprocess, tokenizer=tokenizer, context_length=128)
    # train_dataset = Dataset_load(meta_root='datasets/PH2Derm7pt/', is_train=True,preprocess=preprocess,tokenizer=tokenizer,context_length=256)
    dataset_size_your = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=False)
    # checkpoint = torch.load("model_pt/PubMedCLIP_RN50.pth")
    # net.load_state_dict(checkpoint['state_dict'])
    val_dataset = Dataset_load(meta_root='datasets/Derm7pt/data/', is_train=False,
                               preprocess=preprocess, tokenizer=tokenizer, context_length=128)
    # val_dataset = Dataset_load(meta_root='datasets/PH2Dataset/data/', is_train=False,
    #                            preprocess=preprocess, tokenizer=tokenizer, context_length=128)
    dataset_size_test = len(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
    text = tokenizer(["pigment_network is absent", "pigment_network is typical", "pigment_network is atypical",
                      "streaks is absent", "streaks is regular", "streaks is irregular",
                      "regression_structures is absent", "regression_structures is combinations",
                      "regression_structures is blue areas", "regression_structures is white areas",
                      "dots_and_globules is absent", "dots_and_globules is regular", "dots_and_globules is irregular",
                      "blue_whitish_veil is absent", "blue_whitish_veil is present"], context_length=context_length).to(
        device)
    # text = tokenizer(["pigment_network is absent", "pigment_network is typical", "pigment_network is atypical",
    #                   "streaks is absent", "streaks is present",
    #                   "regression_structures is absent", "regression_structures is present",
    #                   "dots_and_globules is absent", "dots_and_globules is typical", "dots_and_globules is atypical",
    #                   "blue_whitish_veil is absent", "blue_whitish_veil is present"], context_length=context_length).to(
    #     device)

    net = CptAFT(net_biomedCLIP, concept_num=15, class_num=2).to(device)
    # net = CptAFT(net_biomedCLIP, concept_num=12, class_num=2).to(device)
    ###############################################################################################################
    optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-6)
    net.to(device)
    phase = "train"
    model_name = "1_derm7pt_biomedCLIPvitb16"
    ckt_gap = 20
    best_epoch = 0
    best_acc = 0
    n = 0
    # writer = SummaryWriter('log/'+model_name + '/log')
    iter_num=0
    accuracy = 0
    num = 0
    for epoch in range(1, 151):
        net.train()
        total_loss = 0
        batch_num = 0
        accuracy_train = 0
        num_train = 0
        for images, label_tokens, label,concept_lab in train_dataloader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            label = label.to(device)
            concept_lab = concept_lab.to(device)
            # print("image:",images.shape)
            # print("label:", label[0])
            # print("concept_lab:",type(concept_lab[0]) )
            batch_num += 1
            correct_train = 0
            # logits_per_image, logits_per_text = net(images, label_tokens)
            image_features, text_features, logit_scale, decoded_concept, concept_logits_per_image, output= net(images, label_tokens,text,True)
            # image_features, text_features, logit_scale,concept_decoded,concept_logits_per_image,concept_logits_restruct,output = net(images, label_tokens,text,True)
            concept_logits = concept_logits_per_image.softmax(dim=-1)
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            # print("ground_truth:",ground_truth)
            # print("logits_per_image:",logits_per_image)
            # print("logits_per_text:",logits_per_text)
            # 优化器梯度清零
            optimizer.zero_grad()
            recons_loss = h_reconst_criterion(decoded_concept, images.data)
            text_recons_loss = concept_reconst_criterion(concept_logits, concept_lab.float())
            # concept_recons_loss = concept_reconst_criterion(concept_logits, concept_logits_restruct)
            # print("concept_logits:",concept_logits)
            # print("concept_lab:", concept_lab)
            cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            # predicted = torch.max(output, 1)[1]
            pred_loss = prediction_criterion(output, label)
            predicted = torch.max(output, 1)[1]
            correct_train += (predicted == label).sum().cpu().numpy()
            accuracy_train += correct_train / 2 * 100
            num_train += 1
            # loss = 0.4 * cur_loss + 0.4 * pred_loss + 0.2 * recons_loss
            # loss=0.25*cur_loss+0.25*pred_loss+0.25*recons_loss+0.25*text_recons_loss
            loss = cur_loss + pred_loss+ text_recons_loss+ recons_loss #+concept_recons_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            iter_num = iter_num + 1
            # writer.add_scalar('info/recons_loss', recons_loss, iter_num)
            # writer.add_scalar('info/cur_loss', cur_loss, iter_num)
            # writer.add_scalar('info/pred_loss', pred_loss, iter_num)
            # writer.add_scalar('info/loss', loss, iter_num)
            #
            if device == "cpu":
                optimizer.step()
            else:
                optimizer.step()
                # convert_weights(net)
            total_loss += loss
            if batch_num % 32 == 0:
                logger.info('{} epoch:{} loss:{}'.format(phase, epoch, loss))

        print("train_accuracy:", accuracy_train / num_train, "%")
        epoch_loss = total_loss / dataset_size_your

        accuracy = 0
        num=0
        for images, label_tokens,label in val_dataloader:
            correct = 0

            images = images.to(device)
            label_tokens = label_tokens.to(device)
            label = label.to(device)
            # print("label:", labels)
            with torch.no_grad():
                _,output = net(images, label_tokens,text,False)

                predicted = torch.max(output, 1)[1]
                correct += (predicted == label).sum().cpu().numpy()
                accuracy += correct / 1 * 100
            num += 1
        print("val_accuracy:", accuracy / num, "%")
        if accuracy/num > best_acc:
            best_acc = accuracy/num
            best_epoch = epoch
            checkpoint_path = "checkpoint/" + f"{model_name}_best_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)

        logger.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
        print("best_epoch:",best_epoch)
        print("best_acc:", best_acc)
        if epoch % 10 == 0:
            checkpoint_path = "checkpoint/" + f"{model_name}_epoch{epoch}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        logger.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
    print("best_acc:{},best_epoch:{}".format(best_acc, best_epoch))
    # writer.close()
