import torch.nn.functional as F
from torch import nn
import torch
import  numpy as np
class CptAFT(nn.Module):
    def __init__(self, biomedclip,concept_num,class_num):
        super(CptAFT, self).__init__()
        self.biomedclip=biomedclip
        # 添加全连接层
        # self.fc_layer = nn.Linear(concept_num, class_num)
        self.classifier=nn.Sequential(
            nn.Linear(concept_num, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, class_num)
        )
        # self.classifier=nn.Sequential(
        #     nn.Linear(concept_num, 128),
        #     # nn.Tanh(),
        #     nn.Linear(128, 64),
        #     # nn.Tanh(),
        #     nn.Linear(64, 32),
        #     # nn.Tanh(),
        #     nn.Linear(32, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, class_num)
        # )

        self.concept_num=concept_num
        self.dout = int(np.sqrt(224*224) // 4 - 3 * (5 - 1) // 4)
        self.concept = nn.Linear(512, concept_num)
        # Decoding
        self.unlinear = nn.Linear(1, self.dout ** 2)  # b, nconcepts, dout*2
        self.deconv3 = nn.ConvTranspose2d(concept_num, 16, 5, stride=2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5)  # b, 8, (dout -1)*2 + 9
        self.deconv1 = nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1)  # b, nchannel, din, din

    def forward(self, images,label_tokens,text,is_train):
        if is_train:

            image_features, text_features, logit_scale = self.biomedclip(images, label_tokens)
            # print("image_features:", image_features.shape)
            # print("image_features:",image_features.shape)
            # print("text_features:", text_features.shape)
            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # print("concept_logits_per_image:",concept_logits_per_image.shape)

            concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # print("concept_encoded:", concept_encoded.dtype)
            q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # print("q.shape:", q.shape)
            # q = F.relu(self.deconv3(q))
            # q = F.relu(self.deconv2(q))
            # decoded_concept = F.tanh(self.deconv1(q))
            q = self.deconv3(q)
            q = self.deconv2(q)
            decoded_concept = self.deconv1(q)
            # print("decoded_concept.shape:", decoded_concept.shape)

            # output=self.fc_layer(concept_logits_per_image)
            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            output = output.softmax(dim=-1)

            return image_features, text_features, logit_scale,decoded_concept,concept_logits_per_image,output

        else:
            # images=images.cpu()
            # text = text.cpu()
            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # output = self.fc_layer(concept_logits_per_image)

            # concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # # print("concept_encoded:", concept_encoded.dtype)
            # q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # # print("q.shape:", q.shape)
            # # q = F.relu(self.deconv3(q))
            # # q = F.relu(self.deconv2(q))
            # # decoded_concept = F.tanh(self.deconv1(q))
            # q = self.deconv3(q)
            # q = self.deconv2(q)
            # decoded_concept = self.deconv1(q)

            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            output = output.softmax(dim=-1)

            return concept_logits_per_image, output#,decoded_concept

    def register_hook(self, layer):
        def hook(module, input, output):
            self.feature_map_hook = output

        layer.register_forward_hook(hook)

class classification_AE(nn.Module):
    def __init__(self, biomedclip,concept_num,class_num):
        super(classification_AE, self).__init__()
        self.biomedclip=biomedclip
        # 添加全连接层
        # self.fc_layer = nn.Linear(concept_num, class_num)
        self.classifier = nn.Sequential(
            nn.Linear(concept_num, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, class_num)
        )
        # self.classifier=nn.Sequential(
        #     nn.Linear(concept_num, 128),
        #     # nn.Tanh(),
        #     nn.Linear(128, 64),
        #     # nn.Tanh(),
        #     nn.Linear(64, 32),
        #     # nn.Tanh(),
        #     nn.Linear(32, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, class_num)
        # )

        self.decoder = nn.Sequential(
            nn.Linear(class_num, 16),
            # nn.Tanh(),
            nn.Linear(16, 32),
            # nn.Tanh(),
            nn.Linear(32, 64),
            # nn.Tanh(),
            nn.Linear(64, 128),
            # nn.Tanh(),
            nn.Linear(128, concept_num)
        )

        self.concept_num=concept_num
        self.dout = int(np.sqrt(224*224) // 4 - 3 * (5 - 1) // 4)
        self.concept = nn.Linear(512, concept_num)
        # Decoding
        self.unlinear = nn.Linear(1, self.dout ** 2)  # b, nconcepts, dout*2
        self.deconv3 = nn.ConvTranspose2d(concept_num, 16, 5, stride=2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5)  # b, 8, (dout -1)*2 + 9
        self.deconv1 = nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1)  # b, nchannel, din, din

    def forward(self, images,label_tokens,text,is_train):
        if is_train:

            image_features, text_features, logit_scale = self.biomedclip(images, label_tokens)
            # print("image_features:", image_features.shape)
            # print("image_features:",image_features.shape)
            # print("text_features:", text_features.shape)
            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # print("concept_logits_per_image:",concept_logits_per_image.shape)

            concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # print("concept_encoded:", concept_encoded.dtype)
            q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # print("q.shape:", q.shape)
            # q = F.relu(self.deconv3(q))
            # q = F.relu(self.deconv2(q))
            # decoded_concept = F.tanh(self.deconv1(q))
            q = self.deconv3(q)
            q = self.deconv2(q)
            decoded_concept = self.deconv1(q)
            # print("decoded_concept.shape:", decoded_concept.shape)

            # output=self.fc_layer(concept_logits_per_image)
            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            concept_logits_restruct=self.decoder(output)

            output = output.softmax(dim=-1)

            return image_features, text_features, logit_scale,decoded_concept,concept_logits_per_image,concept_logits_restruct,output

        else:
            # images=images.cpu()
            # text = text.cpu()
            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # output = self.fc_layer(concept_logits_per_image)

            # concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # # print("concept_encoded:", concept_encoded.dtype)
            # q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # # print("q.shape:", q.shape)
            # # q = F.relu(self.deconv3(q))
            # # q = F.relu(self.deconv2(q))
            # # decoded_concept = F.tanh(self.deconv1(q))
            # q = self.deconv3(q)
            # q = self.deconv2(q)
            # decoded_concept = self.deconv1(q)

            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            output = output.softmax(dim=-1)

            return concept_logits_per_image, output#,decoded_concept

class classification_copceptonly(nn.Module):
    def __init__(self, biomedclip,concept_num,class_num):
        super(classification_copceptonly, self).__init__()
        self.biomedclip=biomedclip
        # 添加全连接层
        # self.fc_layer = nn.Linear(concept_num, class_num)
        self.classifier = nn.Sequential(
            nn.Linear(concept_num, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, class_num)
        )
        # self.classifier=nn.Sequential(
        #     nn.Linear(concept_num, 128),
        #     # nn.Tanh(),
        #     nn.Linear(128, 64),
        #     # nn.Tanh(),
        #     nn.Linear(64, 32),
        #     # nn.Tanh(),
        #     nn.Linear(32, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, class_num)
        # )

        self.concept_num=concept_num
        self.dout = int(np.sqrt(224*224) // 4 - 3 * (5 - 1) // 4)
        self.concept = nn.Linear(512, concept_num)
        # Decoding
        self.unlinear = nn.Linear(1, self.dout ** 2)  # b, nconcepts, dout*2
        self.deconv3 = nn.ConvTranspose2d(concept_num, 16, 5, stride=2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5)  # b, 8, (dout -1)*2 + 9
        self.deconv1 = nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1)  # b, nchannel, din, din

    def forward(self, images,text,is_train):
        if is_train:

            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # print("concept_logits_per_image:",concept_logits_per_image.shape)

            concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # print("concept_encoded:", concept_encoded.dtype)
            q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # print("q.shape:", q.shape)
            # q = F.relu(self.deconv3(q))
            # q = F.relu(self.deconv2(q))
            # decoded_concept = F.tanh(self.deconv1(q))
            q = self.deconv3(q)
            q = self.deconv2(q)
            decoded_concept = self.deconv1(q)
            # print("decoded_concept.shape:", decoded_concept.shape)

            # output=self.fc_layer(concept_logits_per_image)
            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            output = output.softmax(dim=-1)

            return decoded_concept,concept_logits_per_image,output

        else:
            # images=images.cpu()
            # text = text.cpu()
            class_image_features, class_text_features, class_logit_scale = self.biomedclip(images, text)
            # output = self.fc_layer(logits_image)
            concept_logits_per_image = class_logit_scale * class_image_features @ class_text_features.t()
            # output = self.fc_layer(concept_logits_per_image)

            # concept_encoded = F.relu(concept_logits_per_image).view(-1, self.concept_num, 1)
            # # print("concept_encoded:", concept_encoded.dtype)
            # q = self.unlinear(concept_encoded).view(-1, self.concept_num, self.dout, self.dout)
            # # print("q.shape:", q.shape)
            # # q = F.relu(self.deconv3(q))
            # # q = F.relu(self.deconv2(q))
            # # decoded_concept = F.tanh(self.deconv1(q))
            # q = self.deconv3(q)
            # q = self.deconv2(q)
            # decoded_concept = self.deconv1(q)

            output = self.classifier(concept_logits_per_image.softmax(dim=-1))

            output = output.softmax(dim=-1)

            return concept_logits_per_image, output#,decoded_concept

