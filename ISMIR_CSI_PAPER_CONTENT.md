# ADAPTING COVER SONG IDENTIFICATION METHODS TO THE UNIQUE FEATURES OF INDIAN POPULAR MUSIC

## Abstract
Indian popular music, characterized by its unique scales and vocal styles, presents additional challenges for cover song identification. 
While there are culture-specific char- acteristics in tcovers of Indian popular songs, there are sim- ilar attributes that are culturally invariant. 
We leverage this insight by fine-tuning CQTNet, a state-of-the-art model pre-trained on a large dataset of Western cover songs. We notice an enhancement in results when fine-tuning is per- formed using the Indian popular songs dataset. To mitigate1 the problem of data scarcity, we conducted experiments1 involving data augmentation, creating modified cover ver- sions with altered tempo and tonic. In our training, we investigated the effectiveness of Triplet Loss and its en- hanced version, which takes supplementary positive and negative instances that exhibit varying degrees of similar- ity and dissimilarity. Considering the prominence of the vocal melody in Indian popular songs, we also investigated the effectiveness of separating the vocal from the audio and achieved significant improvement. Additionally, we investigated whether large audio models like HuBERT and MERT can provide any information to improve the result.

## 1 Introduction
Indian popular music, known for its lively melodies and rhythms, is hugely popular across India and many other countries in South Asia, with cover songs becoming a big part as artists rework old and new hits. Identifying these cover songs matters for improving music recommenda- tions and protecting copyrights. Indian popular songs are inspired by the raga-based melodies of Indian classical mu- sic and the stylistic elements of Western music, blending these influences into a distinct musical identity. In the the- ory and practice of Western music, harmony is a core con- cept [1]. Unlike Western songs, Indian popular music gives more importance to the main vocal melody. The vocal- ists generally employ melismatic singing and ornamenta- tions. Apart from these characteristics, most Indian popu- lar songs exhibit a distinct melodic and rhythmic structure.  
The unique features and salient components of Indian popular music play a significant role in the similarity be-
tween original songs and their cover versions, as these el-
ements tend to be retained in the covers. Given the im-
portance of vocal melody in Indian popular music, cover
versions typically retain the original melody with only mi-
nor changes. The design of many cover song identifica-
tion methods has been motivated by the assumptions be-
hind what makes a cover [2]. The prevailing notions of
what constitutes a cover version in Indian popular music
underpin this research inquiry and motivate the need for a
separate investigation.
Earlier cover song identification approaches involved
extracting features from audio signals and measuring the
similarity between songs using these extracted features.
Initial approaches relied on pitch detection and finding
the optimal alignment between the query and the can-
didate song [3]. Quite a good number of approaches
investigated the effectiveness of chroma features, which
represent the energy distribution across different musical
notes [4–6]. Dynamic programming-based approaches like
Dynamic Time Warping (DTW) [7] and Smith-Waterman
alignment [8, 9] enabled the sequence similarity.
Later, deep learning transformed cover song identifica-
tion by automating feature extraction and improving ac-
curacy, outperforming traditional methods. Some of these
approaches aim to learn song representations that are sub-
sequently used to calculate song similarities. Notable ap-
proaches in this direction include the siamese network-
based method [10] and the CNN-based CQTNet [11].
Chang et al. [12] transformed cover song identification
into an image classification task by applying convolutional
neural networks to cross-similarity matrices derived from
songs’ chroma features [13]. ByteCover investigates a fea-
ture learning method with the ResNet model along with
multi-loss training [14]. CoverHunter [15] uses a trans-
former model with attention and a combined focal, center,
and triplet loss function.
There are no culture-specific approaches to cover song
identification in any musical tradition. However, culture-
specific investigations have been conducted in numerous
other music information retrieval tasks. For instance, re-
search has been conducted on pitch detection in Turkish
music [16, 17], Indian popular and classical music [18, 19],
Flamenco singing [20, 21], Arab music [22, 23]. The
culture-specific attributes of music emotion detection have
initiated investigations in Turkish music [24, 25], Indian
popular music [26, 27], Greek music [28].
Our research investigates cover song identification in
Indian popular music, driven by two key questions:
1. Is a specialized model, trained exclusively on Indian
popular music, necessary for effective cover song
identification?
2. Do unique characteristics of Indian popular music
necessitate custom-designed approaches for accu-
rate cover song identification?
Being one of the best and widely experimented cover
song identification methods, we choose CQTNet [11] as
our baseline. Considering our limited dataset of Indian
popular songs and the fact that there are cultural invari-
ant similarity attributes between the original and the cover
version of a song, we experiment with fine-tuning CQTNet
with cover song dataset for Indian popular songs (Cover-
sIndian). The performance gains with the fine-tuned model
over the original CQTNet model provide a clear answer
to the first research question. Leveraging the knowledge
that vocal melody plays a dominant role in Indian popu-
lar songs, we investigate the effectiveness of isolating vo-
cals from the audio signal. This pre-processing step yields
improved results on the CoversIndian dataset, whereas it
lowers the results on the Covers80 dataset [29]. This out-
come answers our second research question, highlighting
the importance of culturally informed approaches.
To address the challenge of limited training data, we
explore a data augmentation technique. Additionally, we
examine the impact of triplet loss [30] and propose a mod-
ification to the triplet loss tailored for CQTNet. While
fine-tuning and vocal separation show significant improve-
ments in results, certain combinations of the other pro-
posed modifications also yield observable enhancements.
To the best of our knowledge, this study is the first to
specifically address cover song identification within the
context of a particular music culture

## 2 Methodology
### 2.1 CQT NET
CQTNet [11] has a CNN-based architecture that processes
Constant-Q Transform (CQT) of the audio as input rep-
resentations. The model effectively captures musical pat-
terns across different pitch ranges through its unique archi-
tecture, featuring dilated convolutions and specialized fil-
ter designs in the initial layers. The model has been bench-
marked on various datasets and attains a maximum MAP
of 0.917 for the YouTube350 and 0.933 for the Mazurkas
dataset. The model is primarily trained on large Western
cover song datasets SHS100K. The model is trained for
the label classification task, where the number of classes
is the number of distinct original songs. The input CQT
representations are passed through the convolution layers
to fully connected Layers, and cross entropy loss is em-
ployed for label classification.
During inference, the embedding for a song is taken
from the penultimate layer. The distance between the
songs is computed by taking the cosine distance between
the embeddings. The songs are ranked according to their
cosine distance computed from the previous step. Consid-
ering the performance of CQTNet for cover song identifi-
cation, we base our explorations on this model.

### 2.2 Fine-Tuning with Indian Cover Songs Dataset
Certain factors in cover song identification remain consis-
tent across cultures. The original CQTNet model, trained
on an extensive Western cover song dataset, has learned to
identify the key factors that distinguish cover songs. Fine-
tuning the model on the CoversIndian dataset enables tar-
geted adaptation to Indian cover songs' cultural-musical
attributes while maintaining transfer learning benefits from
training on Western music. This is particularly important
given the limited number of labeled data on Indian cover
songs. In all our further experiments, we use the CQTNet
model, which is fine-tuned with our CoversIndian dataset.
Given the low number of classes to classify in the Indian
songs dataset, we reduced the number of nodes in the final
classification layer of the architecture.

### 2.3 Vocal Separation
Indian popular music is heavily influenced by Indian clas-
sical music and Western music. Even though the propor-
tion of the blend is not generally uniform, the song's iden-
tity primarily lies in the predominant vocal melody. This
suggests that the vocal component plays a more significant
role in determining similarity. The importance of vocal
melody motivated the experiments with extracting vocals
and ignoring the accompanying instruments. Our experi-
ments demonstrate that vocal separation is especially ben-
eficial for Indian music, where vocal nuances are a key
distinguishing factor.

### 2.4 Data Augmentation

The limited dataset and availability of only one cover song
per original song pose challenges in learning diverse pos-
sible variations in a cover version of an original song. To
compensate for this, we applied a basic data augmentation
technique on the audio dataset. Feng et. al. [15] have ex-
perimented with data augmentation for cover song identi-
fication. We created 4 variants of a cover song with two
pitch variations of +4 and -3 semitones, and two tempo
variations of 1.2x and 0.8x were applied on the cover ver-
sion. This results in a total of 5 cover versions for each
original song in the training data. This would help the
model to learn similarity with more variants.

### 2.5 Triplet Loss
When the cross-entropy loss helps in training the model to
recognize similarities with a positive cover song, the triplet
loss enables it to learn the dissimilarities with a non-cover
song of the original song. Triplet loss minimizes the dis-
tance between anchor-positive pairs while maximizing the
distance between anchor-negative pairs by a margin. The
loss L is computed by taking the maximum between the
distance(Anchor-Positive, Anchor-Negative) and 0, ensur-
ing that the loss remains non-negative.

L = max (0, a + d(Aori, Pcover(i)) - d(Aori, Nrand))

where d() is the distance function, Aori represents the orig-
inal song (anchor) embedding, Pcover(i) denotes the i-th
cover of the song, Nrand is a random negative example
and o is the margin of separation between the positive and
negative examples. Triplet loss has been experimented in
cover song identification before in CoverHunter [15] and
ByteCover [14].

### 2.6 Modified Triplet Loss Function
To further refine our embedding space, we modified the
triplet Loss function to focus on the most challenging ex-
amples. It is very likely to have negative cover instances
which are similar to the original song. The strong melody
prominence in Indian popular songs increases the likeli-
hood of negative cover instances closely resembling the
positive cover instances of the original song. This modi-
fied loss encourages the network to learn more discrimina-
tive features by considering the subtle differences with the
most similar negative examples. The modified triplet loss
is defined as:

L = max {0, @+2d(Aori, Pcover(i)) -[d(Aori, Nsim) +d(Aori, Nrand)]}

where Aori represents the original song (anchor) embed-
ding, Pcover(i) denotes the i-th cover of the song, Nsim is
the most similar negative example, Nrand is a random neg-
ative example and o is the margin of separation between
the positive and negative examples. We select the nega-
tive example exhibiting the smallest distance to the anchor
along with a random negative example. We incorporate
a weighting factor of 2 applied to the distance between
anchor-positive instances. This balances the contributions
of the positive and negative terms. As a result, the model
achieves improved generalization across both general and
edge-case negative examples.
The experiments with the modified triplet loss function
involve two fine-tunings. The representation obtained from
the model after the first fine-tuning is utilized to identify
the most similar negative cover instance for each original
song in the training data.
Algorithm2.6 describes the fine-tuning with the modi-
fied triplet loss. M is the model from the first fine-tuning.
This model is used to compute the embeddings for the orig-
inal and cover songs in the train data. Aorig represents the
anchor embedding for each original song s and Cover A is
one of the cover songs of Aprig; Nrand is a random negative
example; and Nsim is the most similar negative example.

### 2.7 Combining CQTNet and MERT/HuBERT
Embeddings
Large audio representations are found helpful for various
downstream tasks involving audio data. We investigate the

Algorithm 1 Modified Triplet Loss Fine-tuning
1: M <- Fine-tune on label classification.
2: Fine-tune embeddings using Modified Triplet Loss:
3: E = get_embeddings(M, D), where D is the dataset with instances as tuples (Dorig, Dcover)
4: for each Aorig, CoverA in E do 
5: CA <- All covers of Aorig
6: Nrand := random(Ecover), where Nrand ¢ CA
7: Nsim := argmin Dist(Aorig, Ecover(i)), s.t Ecover(i)¢C4
8: L := Triplet(Aorig, Cover A, Nrand + Nsim)
9: end for
10: Update model parameters based on the computed loss.


potential of large audio models to gain insights into the
relationship between a song and its cover version.
HuBERT [31] (Hidden unit BERT) is a state-of-the-art
approach for self-supervised speech representation learn-
ing in tasks like speech recognition, generation, compres-
sion, and speaker verification in both clean and noisy con-
ditions. MERT [32](Music undERstanding model with
large-scale self-supervised Training) adapts this frame-
work specifically for music audio. It tries to address the
unique challenges of representing the tonal and pitched
characteristics of musical content. MERT has demon-
strated benchmarking performance across different Mu-
sic Information Retrieval tasks like beat tracking, singer
identification, music tagging, key detection, genre classifi-
cation, and more. Both models leverage self-supervised
learning to create rich audio representations without re-
quiring labeled data. This makes them powerful founda-
tions for various downstream audio analysis applications.
We experimented with an approach that concatenates
CQT features with MERT embeddings. We aimed to deter-
mine whether the complementary information captured by
MERT/HuBERT can enhance cover song similarity when
combined with the traditional CQT features. The MERT
embeddings are computed using the pre-trained model
(without updating during training) and are passed through
feed-forward layers to capture only the relevant features.
We obtain the embeddings from CQTNet by processing the
CQT features of the same audio sample. The embeddings
from the MERT feed-forward layer and the CQTNet model
are combined and then passed through a final feed-forward
layer. Figure 1 describes the architecture of this approach.
Cosine similarity is employed during the inference.

## 3. EXPERIMENTS

### 3.1 Dataset

The CoversIndian dataset comprises audio data from orig-
inal songs and their covers. The dataset contains 245 song
pairs from 2 different Indian languages: Hindi and Malay-
alam. Of these, 126 song pairs are for fine-tuning, and 119
song pairs are for Testing.
We have also used Covers80 dataset [29] as test set
for showing certain results on a Western music dataset. It

Embeddings

Feed-Forward NN

Combining Feed-
Forward NNS

MERT
HuBERT

Feed-Forward NN

Convolutional Layers

CQTNet

Audio

CQT Features

Figure 1. CQTNet+MERT/HuBERT Architecture

contains 80 pairs of Western songs.

### 3.2 Experimental Setup
We computed the CQT representations using the Librosa
library [33] for the audio datasets to train and evaluate
the models. Two sets of data preprocessing were applied
on the CoversIndian and WesternCovers datasets, as men-
tioned in the Methodology sections 2.3 and Section 2.4.
Vocal Separation from the datasets were done using the
API from music.ai (erstwhile Moises) [34]. We compute
CQT representations of the vocal-separated audio to train
and evaluate the models.
Data Augmentation techniques were applied to the cover
versions of both the audio and vocal-separated audio
datasets. Data augmentation can result in generating 4 ad-
ditional versions for each cover song, varying in pitch and
tempo. The data augmentations of pitch shift and tempo
variations, as discussed in Section 2.4, were applied using
the Librosa library. We compute the CQT representations
for all the augmented data and use them for training and
analysing the models.
Experiments were carried out on Intel Gaudi3 AI accel-
erators (128GB) and an NVIDIA A100 GPU (80GB).
For reproducibility and consistency across all experiments,
we conducted the experiments in Random Seed(0) envi-
ronment. All label classifications were fine-tuned for 200
epochs, and all Triplet Loss fine-tunings for 150 epochs.
A batch size of 60 was used with shuffle set to False; This
helps to process 10 sets of 6 song versions, 1 original and
5 augmented covers for Triplet Loss fine-tuning.

### 3.3 Evaluation

We used Mean Average Precision (MAP), Precision@10
(Top10), and mean rank of the first correctly identified
cover (Rank1) as our evaluation metrics. All the metrics
are widely used for Cover Song identification problems.

## 4. RESULTS

The following acronyms have been used in the results ta-
bles.

	
Acronym	        | Details
VocalSep	    | Vocal Separated Audio
DA              | Data Augmentation
Trip.Loss	    | Triplet Loss
Mod Trip.Loss   | Modified Triplet Loss

Table 1. Acronym Reference

### 4.1 Baseline CQTNet Vs Fine-tuned CQTNet
This experiment fine-tunes the original CQTNet model
pre-trained exclusively on Western cover songs to the Cov-
ersIndian dataset.

	
| Model              |	MAP  |	Top10 |	Rank1 |
| Baseline  CQTNet   |0.7238 | 0.0840 |  8.87 |
| Fine-tune CQTNet	 | 0.7379|0.0819  | 11.53 |

Table 2. Results of baseline CQTNet Vs CQTNet fine-
tuned with CoversIndian


## 4.2 Vocal Separation
In this experiment, we try to see the impact of vocal sepa-
ration on CoversIndian and Covers80 test set. As reported
by Zhesong et. al. [11], the baseline CQTNet gives high
MAP on Western covers testset Covers80 with source au-
dio. The baseline CQTNet does not give similar results on
CoversIndian test set. Results are shown in Table 3. As
hypothesized, vocal separation yielded a notable improve-
ment for CoversIndian. We can see this improvement on
CoversIndian test set in both the experiments using base-
line CQTNet and fine-tuned CQTNet. However, there is
a degradation in MAP for Covers80. The results strongly
back the claim that Indian songs are predominantly based
on vocal melody. However, it is not significant to Western
music.

| Dataset|	Method|	MAP|	Top10|	Rank1|
| CoversIndian|	Baseline CQTNet|	0.7238|	0.0840|	8.87|
|   | 	Baseline CQTNet + VS|	0.8629|	0.0937|	3.95|
|	|Fine-tuned CQTNet|	0.7379|	0.0819|	11.53|
|	|Fine-tuned CQTNet + VS|	0.9001|	0.0962|	2.51|
|Covers80	Baseline CQTNet|	0.8768|	0.0944|	3.30|
|	|Baseline CQTNet + VS|	0.7181|	0.0819|	8.79|


Table 3. Results with vocal separation

The vocal separated pre-processed audio consistently
led to better performance compared to source audio in-
puts. Notably, similar experiments conducted on West-
ern cover songs demonstrated that vocal separation led to
lower MAP values compared to their original cover songs,
underscoring that Indian cover songs inherently emphasize
vocal nuances- which in turn contribute to the improved
MAP observed.

### 4.3 Data Augmentation
CQTNet was fine-tuned with the Augmented CoversIndian
dataset. Table 5 shows the results with data augmentation.

|Method|	MAP|	Top10|	Rank1|
|VocalSep| 0.9001 |0.0962 |2.51 |
|DA+VocalSep| 0.9131 | 0.0958|2.91|
|Trip.Loss+VocalSep| 0.8839|0.0979|3.01|
|Trip.Loss+DA+VocalSep| 0.8953|	0.0971|	3.33|
|Mod Trip.Loss+VocalSep|0.8922|0.0962|2.14|
|Mod Trip.Loss+DA+VocalSep|	0.9234|	0.0975|	2.27|

Table 4. Consolidated results on CoversIndian test set. Mentioned modifications applied on fine-tuned model with Cover-
sIndian train set. (Refer Table 4 for acronyms used).

The results can be compared againt Table 3. We can see
that the data augmentation gives improvement on vocal
separated data, but gives no improvement on source audio
data.

| Method| MAP| Top10 | Rank1 |
| DA + Source Audio| 0.7307| 0.0857| 9.89 |
| DA + VocalSep Audio| 0.9131| 0.0958| 2.91|

Table 5. Results after applying data augmentation on Cov-
ersIndian testset


### 4.4 Triplet Loss Function
We used the Triplet Loss function to analyse how well it
captures the variations in cover song versions of source
and vocal-separated audio. In this experiment, we use the
Triplet Loss function for the penultimate embedding layer
instead of label classification using Cross Entropy at the
last layer. The results in Table 6 show that enabling triplet
loss gives a significant improvement on the source audio
data with data augmentation. However, on data augmented
vocal separated triplet loss leads to performance degrada-
tion

Method          MAP Top10 Rank1
DA + Source Audio  0.7696 0.0861  6.97
DA + VocalSep Audio 0.9079 0.0975 2.56

Table 6. Results with triplet loss on CoversIndian test set




### .5 Modified Triplet Loss
As mentioned in the 2.6, this involves 2 levels of fine-
tuning to get the most similar negative cover instance for
each original song in the train set.

| Method| MAP| Top10| Rank1 |
| DA + Source Audio| 0.7901| 0.0899| 5.83 |
| DA + VocalSep Audio| 0.923| 0.0975| 2.2 |


Table 7. Results with modified triplet loss on CoversIndian
test set.

The Modified Triplet Loss function showed promising
improvements on the Indian cover song datasets. The
result in Table 7 suggests that the Modified Triplet loss
helps the embeddings to capture the subtle differences with
the negative cover instances. Our comparative evaluation
demonstrates that the Modified Triplet Loss outperforms
Triplet Loss in capturing variations in augmented cover
songs on source audio and vocal separated data.

4.6 Additional Evaluation of HuBERT and MERT
Table 8 summarizes our experiments with MERT330M
 and the HuBERT-large model.
We made the MERT/HuBERT weights non-trainable during the exper-
iment and passed the pre-trained representations to CQT-
Net as detailed in Section 2.7. No improvements were seen
in these experiments, so we did not explore this direction
further. However, merging embeddings based on attention
can be tried in future.

|Model on CoversIndian|MAP|Top10|Rank1|
|CQTNet+MERT|0.5017|0.0664|28.03|
|CQTNet+Hubert|0.4466|0.0601|37.60|


Table 8. Results with architecture combining
MERT/HuBERT with CQTNet

5. CONCLUSION

This study successfully adapts a state-of-the-art cover song
identification method CQTNet, to the unique features of
Indian popular music, addressing challenges posed by its
distinct melodic and vocal characteristics. By fine-tuning
CQTNet on the CoversIndian dataset, we demonstrate
significant performance improvements, particularly when
leveraging vocal separation techniques that emphasize the
prominence of vocal melodies in Indian music. Data aug-
mentation further mitigates the issue of limited training
data, enabling the model to learn from diverse variations
of cover songs.
The introduction of a modified triplet loss function
proves effective in capturing subtle differences between
similar negative cover instances and positive covers, en-
hancing the model's ability to discriminate between chal-
lenging examples. While large audio models like HuBERT
and MERT did not yield improvements in this context, fu-
ture exploration may unlock their potential.
Our findings highlight the importance of culturally in-
formed pre-processing steps and loss functions tailored to
specific musical traditions. This research lays the ground-
work for further investigations into adapting cover song
identification to diverse musical genres and traditions.