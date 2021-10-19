# Attribute Prediction via Landmark-Upsampling and RoI Pooling with Focal Loss Augmentation

Fashion landmarks defined on clothes provides crucial
information to understand an image representation. With
the aid of effective landmark localization, the precision of
attribute prediction also improves. However, detection of
these landmarks is difficult due to the presence of variations
in human poses and background clutters. Therefore, to address these problems, this work proposes a novel method to
effectively localize landmarks using a multi-label learning
network that employs category and attribute classification.
By introducing transposed convolution to upsample the feature maps along with modified RoI pooling and using spatial attentive module to fine-grained image features, the results show that the model is able to localize the landmarks
more effectively and boosts the predictive performance of
classifying the attributes. Further, different experiments
show that focal loss under some specific settings performs
2.2% times better in accuracy than standard cross entropy
loss on the DeepFashion dataset for the specified task.


Best accuracy achieved with finetuned ResNext-101 model with spatial attention and RoI based transposed convolution landmark upsampling.

Run :

python3.6 train_resnet_101_roi.py
