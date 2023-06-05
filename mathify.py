#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Dataset


# In[10]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# In[11]:



train_df= pd.read_csv('my_dataset.csv')
train_df 


# In[12]:


train_df['body'][0]


# In[13]:


train_df.duplicated()


# In[14]:


train_df.duplicated().sum()


# In[15]:


train_df.describe()


# In[16]:


train_df.info()


# In[17]:


train_df.isna().sum()


# # ***Data Preparation***

# ### 1. Removing punctuations

# In[18]:


# re :  regular expression support.
import re


# In[19]:


train_df['clean_text1'] = train_df['body'].map(lambda x: re.sub('[,\.! ? ]()', ' ', x))
train_df['clean_text2'] = train_df['clean_text1'].map(lambda x: re.sub("'s", 'is', x))
train_df['clean_text3'] = train_df['clean_text2'].map(lambda x: re.sub("'m", 'am', x))
train_df['clean_text'] = train_df['clean_text3'].map(lambda x: re.sub("'re", 'are', x))


# In[15]:


train_df['clean_text'][55]


# In[16]:


train_df['clean_text']


# In[17]:


train_df['clean_text'] 


# ### 2. Lowercase all text:

# In[18]:


train_df['clean_text'] = train_df['clean_text'].map(lambda x: x.lower())


# In[19]:


train_df['clean_text'].sample(10)


# In[ ]:





# ### 3. Remove HTML characters

# In[20]:


html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
train_df['clean_text'] = train_df['clean_text'].apply(lambda x: re.sub(html,'', x))


# In[21]:


train_df['clean_text'].sample(10)


# ### 4. Remove \n

# In[22]:


train_df['clean_text'] = train_df['clean_text'].apply(lambda x: re.sub('\n','', x))


# In[23]:


train_df['clean_text'].sample(10)


# In[ ]:





# ### 5. Remove Non-English characters

# In[24]:


train_df['clean_text'] = train_df['clean_text'].apply(lambda x: re.sub(r'[^\x00-\x7f $ &]',r'', x))


# In[25]:


train_df['clean_text']


# # ***Data Modeling and Tokenization***

# ## 1. Removing stop words and Lemmatization

# ### 1.1 Using gensim NLTK library for removing stop words

# In[26]:


import gensim
from gensim.utils import simple_preprocess
import nltk


# In[27]:


nltk.download('stopwords')


# In[28]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


# In[29]:


import pandas as pd
import nltk
import multiprocessing
from nltk.corpus import stopwords

# Load dataset column into a pandas DataFrame


# Define number of CPU cores to use for parallel processing
num_cores = multiprocessing.cpu_count()


# In[30]:


train_df['clean_text'][0]


# In[31]:


import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
# Load stop words from NLTK library
stop_words = stopwords.words('english')
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'about', 'against', 'between', 'into', 'through', 'during', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Your large dataset, stored in a list called "data"
data = train_df['clean_text']
# Remove stop words from the 'clean_text' column in the DataFrame
train_df['clean_text'] = train_df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


# In[32]:


print(stopwords.words('english'))


# In[33]:


train_df['clean_text'][0]


# In[34]:


text_to_list = train_df.clean_text.values.tolist()


# In[35]:


text_to_list


# In[36]:


len(text_to_list)


# In[37]:


# If you set deacc=True which will removes the punctuations (that we already removed)
def convert_sentences_to_words(sentences):
    for sentence in sentences:        
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# In[38]:


text_as_words = list(convert_sentences_to_words(text_to_list))


# In[39]:


len(text_as_words)


# In[40]:


print(text_as_words[1])


# In[41]:


len(text_as_words[1])


# # 2. Creating Bigram *(2 words compound words)* and Trigram *(3 words compound words)*

# In[37]:


# Note if you will use higher threshold, which will return the fewer phrases.
bigram = gensim.models.Phrases(text_as_words, min_count=5, threshold=100) 
trigram = gensim.models.Phrases(bigram[text_as_words], threshold=100) 


# In[38]:


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[39]:


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

##3. Lemmatization
# In[40]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# ##4. Using Spacy library for removing stop words

# In[41]:


# Importing spacy and Loading model
import spacy

nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])


# In[42]:


# Form Bigrams
clean_words_bigrams = make_bigrams(text_as_words)


# In[43]:


# Do lemmatization keeping only noun, adj, vb, adv
clean_words_lemmatized = lemmatization(clean_words_bigrams, allowed_postags=['NOUN', 'VERB','adj','adv'])


# In[129]:


#pip install nlp


# In[44]:


print(clean_words_lemmatized)


# # 5. Tokenizing the clean and lemmatize words

# In[45]:


print(len(text_as_words))
print(len(text_as_words[0]))
print(len(text_as_words[99]))


# In[46]:


#input liste de liste de tokens
def pos_tagging_nltk(tokens):
    return [nltk.pos_tag(w) for w in tokens if w]


# In[47]:


pos_tagging_nltk(clean_words_lemmatized)


# # Image

# In[ ]:


(////////////////////////////////////)


# In[ ]:


get_ipython().system(' git clone https://github.com/microsoft/fluentui-emoji')


# In[32]:


#pip install pyttsx3


# In[17]:


#pip install playsound


# # /////////////////////////////////GAN

# In[2]:


import os
print(os.listdir("fluentui-emoji/assets"))


# In[11]:


#pip install torchvision


# In[18]:


#pip install torchtext


# In[9]:


#pip install torch


# In[2]:


from torchvision import transforms


# # Image Preprocessing

# In[68]:


from torchtext import datasets
from torchvision import datasets
import torch


# Now you can use the datasets module
batch_size = 32
batchSize = 64
imageSize = 64

# 64x64 images!
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder('fluentui-emoji/assets/', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size)

imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)


# In[21]:


import matplotlib.pyplot as plt

for i in range(10):
    plt.imshow(imgs[i])
    plt.show()


# # Weights

# In[20]:


def weights_init(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# # Generator

# In[21]:


import torch.nn as nn

class G(nn.Module):
    def __init__(self):
        # Used to inherit the torch.nn Module
        super(G, self).__init__()
        # Meta Module - consists of different layers of Modules
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
                nn.Tanh()
                )
        
    def forward(self, input):
        output = self.main(input)
        return output

# Creating the generator
netG = G()
netG.apply(weights_init)


# # Discriminator

# In[24]:


# Defining the discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, input):
        output = self.main(input)
        # .view(-1) = Flattens the output into 1D instead of 2D
        return output.view(-1)
    
    
# Creating the discriminator
netD = D()
netD.apply(weights_init)


# # /////////////////setup

# In[22]:


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()
        
        self.nz = nz
        self.channels = channels
        
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0), # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img

    
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits)
    
        return out.view(-1, 1)


# # Training

# # Initialize models and optimizers

# In[119]:


batch_size = 32
#learning rates for the generator and discriminator
LR_G = 0.001
LR_D = 0.0005

beta1 = 0.5
epochs = 100

real_label = 0.9
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[120]:


netG = Generator(nz).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, 1, 1, device=device)

G_losses = []
D_losses = []
epoch_time = []


# In[121]:


def plot_loss (G_losses, D_losses, epoch):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss - EPOCH "+ str(epoch))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[123]:


def show_generated_img(n_images=5):
    sample = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        sample.append(gen_image)
    
    figure, axes = plt.subplots(1, len(sample), figsize = (150,150))
    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample[index]
        axis.imshow(image_array)
        
    plt.show()
    plt.close()


# # Training Loop

# Loss_D:The loss function is a measure of how well the discriminator is able to distinguish between real and generated images.
# 
# Loss_G:The loss function is a measure of how well the generator is able to produce realistic images.

# In[124]:


import time
from tqdm import tqdm
from torch.utils.data import DataLoader

for epoch in range(epochs):
    
    start = time.time()
    for ii, (real_images, train_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size, 1), real_label, device=device)

        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if (ii+1) % (len(train_loader)//2) == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch + 1, epochs, ii+1, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
    plot_loss (G_losses, D_losses, epoch)
    G_losses = []
    D_losses = []
    if epoch % 10 == 0:
        show_generated_img()

    epoch_time.append(time.time()- start)
    
#             valid_image = netG(fixed_noise)


# The discriminator's loss (0.6696) is relatively high compared to the generator's loss (2.9441). This suggests that the discriminator is able to easily distinguish between real and generated images, but the generator may be struggling to produce realistic images.
# 
# A high value of D(x) indicates that the discriminator is correctly identifying real images.
# A low value of D(G(x)) indicates that the discriminator is correctly identifying the generated images as fake.
# 
# The generator aims to generate images that are increasingly difficult for the discriminator to differentiate from real images by maximizing the value of D(G(x)). 

# In[126]:


import numpy as np

print (">> average EPOCH duration = ", np.mean(epoch_time))


# In[128]:


show_generated_img(10)


# This code generates a batch of generated images using the trained generator network and saves them as image files in a specified directory.

# In[140]:


from PIL import Image
import torchvision.utils as vutils

if not os.path.exists('../output_images'):
    os.mkdir('../output_images')
    
im_batch_size = 50
n_images=10000

for i_batch in tqdm(range(0, n_images, im_batch_size)):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = netG(gen_z)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        vutils.save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


# In[141]:


fig = plt.figure(figsize=(25, 16))
# display 10 images from each class
for i, j in enumerate(images[:32]):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    plt.imshow(j)


# In[142]:


import shutil
shutil.make_archive('images', 'zip', '../output_images')


# In[143]:


torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')


# In[157]:


get_ipython().system('dir ..\\output_images')


# # //////////////Evaluation

# In[69]:


import numpy as np

# Extract a batch of images from the train_loader
imgs, labels = next(iter(train_loader))

# Convert the tensor to a NumPy array and transpose the dimensions
imgs = imgs.numpy().transpose(0, 2, 3, 1)

# Rescale the pixel values from [-1, 1] to [0, 1]
real_samples = (imgs + 1) / 2


# In[70]:


import torch

# Load the saved GAN model
netG = Generator()
netG.load_state_dict(torch.load('generator.pth'))

# Generate a batch of random noise vectors
batch_size = 32
noise = torch.randn(batch_size, 100, 1, 1)

# Generate fake images from the noise vectors using the generator
fake_samples = netG(noise).detach().numpy()

# Rescale the pixel values from [-1, 1] to [0, 1]
fake_samples = (fake_images + 1) / 2


# In[71]:


import numpy as np

def precision_recall(real_samples, generated_samples, threshold=0.5):
    # Compute precision and recall
    true_positives = np.sum(generated_samples >= threshold)
    false_positives = np.sum(generated_samples < threshold)
    false_negatives = np.sum(real_samples >= threshold) - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    # Compute F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
# Example usage
real_samples = np.random.rand(1000)
generated_samples = np.random.rand(1000)
precision, recall, f1_score = precision_recall(real_samples, generated_samples, threshold=0.5)
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1_score))


# These results indicate that the model has a high recall (i.e., it correctly identifies most of the true positive samples) but a low precision (i.e., it also identifies many false positive samples). The F1 score indicates that the model has a moderate performance in terms of both precision and recall.

# # /////////////////////////////////////Text

# # Simple RNN
# 

# In[42]:


import tensorflow as tf

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# Writing a function for getting auc score for validation
# 
# 

# In[43]:


from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = roc_curve(target, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc


# In[44]:


from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(train_df['clean_text'].values,train_df['label'].values, 
                                                  stratify=train_df['label'].values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)


# In[45]:


train_df['clean_text'].apply(lambda x:len(str(x).split())).max()


# In[46]:


from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping


# In[47]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence

# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 100

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index


# In[48]:


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

with strategy.scope():
    # A simpleRNN without any pretrained embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()


# In[49]:


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync) #Multiplying by Strategy to run on TPU's


# In[50]:


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


# In[51]:


loss, accuracy = model.evaluate(xvalid_pad, yvalid, verbose=0)
print('Accuracy: %.2f%%' % (accuracy*100))


# In[52]:


scores_model = []
scores_model.append({'Model': 'SimpleRNN','AUC_Score': roc_auc(scores,yvalid)})


# # ////////////////Word Embeddings

# In[53]:


# load the GloVe vectors in a dictionary:
from tqdm import tqdm

embeddings_index = {}
f = open('glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[54]:


from tqdm import tqdm
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# # LSTM

# In[55]:


from tensorflow.keras.layers import LSTM

with strategy.scope():
    
    # A simple LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
model.summary()


# In[56]:


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


# In[57]:


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


# In[58]:


loss, accuracy = model.evaluate(xvalid_pad, yvalid, verbose=0)
print('Accuracy: %.2f%%' % (accuracy*100))


# In[59]:


scores_model.append({'Model': 'LSTM','AUC_Score': roc_auc(scores,yvalid)})


# # GRU

# In[60]:


from keras.layers import SpatialDropout1D
from keras.layers import GRU

with strategy.scope():
    # GRU with glove embeddings and two dense layers
     model = Sequential()
     model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
     model.add(SpatialDropout1D(0.3))
     model.add(GRU(300))
     model.add(Dense(1, activation='sigmoid'))

     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    
model.summary()


# In[61]:


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


# In[62]:


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


# In[63]:


loss, accuracy = model.evaluate(xvalid_pad, yvalid, verbose=0)
print('Accuracy: %.2f%%' % (accuracy*100))


# In[64]:


scores_model.append({'Model': 'GRU','AUC_Score': roc_auc(scores,yvalid)})


# # Bi-Directional RNN's

# In[65]:


from tensorflow.keras.layers import Bidirectional
from keras.layers import LSTM

with strategy.scope():
    # A simple bidirectional LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    
model.summary()


# In[66]:


model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


# In[67]:


scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))


# In[68]:


loss, accuracy = model.evaluate(xvalid_pad, yvalid, verbose=0)
print('Accuracy: %.2f%%' % (accuracy*100))


# In[69]:


scores_model.append({'Model': 'Bi-Directional RNN','AUC_Score': roc_auc(scores,yvalid)})


# # //////////////////////Evaluation

# In[70]:


scores_model


# In[71]:


# Visualization of Results obtained from various Deep learning models
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
results.style.background_gradient(cmap='Blues')


# In[ ]:


(////////////////////////////)


# In[1]:


import spacy
from spacy import displacy


# In[2]:


import spacy
import requests
import json
from IPython.core.display import HTML, display
import base64

nlp = spacy.load('en_core_web_sm')

def display_image(query):
    # Unsplash API configuration
    ACCESS_KEY = '*******'
    SECRET_KEY = '********'

    # get authorization token
    auth_endpoint = 'https://unsplash.com/oauth/token'
    auth_response = requests.post(auth_endpoint, {
        'grant_type': 'client_credentials',
        'client_id': ACCESS_KEY,
        'client_secret': SECRET_KEY,
    })
    auth_token = json.loads(auth_response.text)['access_token']

    # get image url from Unsplash API
    headers = {'Authorization': f'Bearer {auth_token}'}
    search_endpoint = f'https://api.unsplash.com/search/photos?query={query}'
    search_response = requests.get(search_endpoint, headers=headers)
    image_url = json.loads(search_response.text)['results'][0]['urls']['regular']

    # display image
    img_data = requests.get(image_url).content
    img_html = f'<img src="data:image/png;base64,{base64.b64encode(img_data).decode("utf-8")}" style="width:150px;height:150px;">'
    return img_html


def display_sentence_with_images(sentence):
    doc = nlp(sentence)
    html_elements = []
    for token in doc:
        if token.pos_ == "NOUN":
            img_html = display_image(token.text)
        elif token.pos_ == "nsubj":
            img_html = display_image("person")
        else:
            img_html = ""

        if token.head.i < token.i:
            html_element = f"{img_html} {token.text}"
        else:
            html_element = f"{token.text} {img_html}"
        html_elements.append(html_element)

    html_string = " ".join(html_elements)
    display(HTML(html_string))
    displacy.render(doc, style="dep")

# Example usage
display_sentence_with_images('5 toys')


# In[ ]:





# In[ ]:


import spacy
import requests
import json
from IPython.core.display import HTML, display
import base64
import openai 

nlp = spacy.load('en_core_web_sm')

def display_image(query):
    # Unsplash API configuration
    OPENAI_KEY='******'

    # get authorization token
    #auth_endpoint = 'https://unsplash.com/oauth/token'
    auth_response = requests.post(auth_endpoint, {
        'grant_type': 'client_credentials',
        'client_secret': OPENAI_KEY,
    })
    auth_token = json.loads(auth_response.text)['access_token']

    # get image url from Unsplash API
    headers = {'Authorization': f'Bearer {auth_token}'}
    search_endpoint = f'https://api.unsplash.com/search/photos?query={query}'
    search_response = requests.get(search_endpoint, headers=headers)
    image_url = json.loads(search_response.text)['results'][0]['urls']['regular']

    # display image
    img_data = requests.get(image_url).content
    img_html = f'<img src="data:image/png;base64,{base64.b64encode(img_data).decode("utf-8")}" style="width:150px;height:150px;">'
    return img_html


def display_sentence_with_images(sentence):
    doc = nlp(sentence)
    html_elements = []
    for token in doc:
        if token.pos_ == "NOUN":
            img_html = display_image(token.text)
        elif token.pos_ == "nsubj":
            img_html = display_image("person")
        else:
            img_html = ""

        if token.head.i < token.i:
            html_element = f"{img_html} {token.text}"
        else:
            html_element = f"{token.text} {img_html}"
        html_elements.append(html_element)

    html_string = " ".join(html_elements)
    display(HTML(html_string))
    displacy.render(doc, style="dep")

# Example usage
display_sentence_with_images('5 toys')


# In[ ]:





# In[ ]:




