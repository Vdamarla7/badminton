# Badminton Video Analysis

## Purpose

Badminton is as much a game of anticipation and preparation as it is a game of strength and speed. Understanding your opponent's playing style and favorite shots can give you the split-second advantage you need to win the point. The traditional way of getting this advantage was to observe your opponent's shot selection throughout a tournament and prepare for your match. However, today, data on every professional player on freely available on YouTube. By using modern computer vision techniques, machine learning, and statistics, players can drastically improve their preparation and training. As someone who loves the game of badminton and is looking for ways to improve my game, I aim to use computer vision to improve not only my games but also others. I believe that machine learning will have a significant impact on every sport and can be used to help us reach our physical potential.

In this repo, I will share datasets, code samples, and ML models that can be used to statistically analyze badminton games. All the data sets, code samples, and ML models are free so anyone can freely build their own methods on these datasets.

## Poses

Pose detection is a computer vision technique that identifies and tracks key points on a human body to estimate posture and movement using machine learning. I was inspired by Paul Liu’s work found here: https://cs.stanford.edu/people/paulliu/badminton/.

## VideoBadminton

VideoBadminton is a dataset of badminton clips that can be used to train ML models for shot classification. This data set contains over 7000 videos and contains hundreds of videos of 14 different shots. More information about this dataset may be found here: https://arxiv.org/html/2403.12385v1

I decided to create a derived dataset by extracting the poses and the bounding boxes of the players in this dataset. These poses can then be used to train ML models that utilize poses to classify shots. I am publishing this dataset for free so that anyone can work with these poses without having to extract them themselves.

## How I created the data set:
The code is available here: [extract_poses_with_sapiens.py](https://github.com/Vdamarla7/badminton/blob/main/badminton/extract_poses_with_sapiens.py)
. I used the YOLO model to find the bounding boxes of every person in every single frame. Then I take the two biggest bounding boxes to run the Sapiens Pose Estimation model on, and I draw these onto the frame. The technique of using the two biggest bounding boxes works well for the VideoBadminton dataset, but may not work well with others, as one of the non-players may have a big bounding box.

Unfortunately, this pipeline gave me a few problems as I found that some videos in the dataset could not be read, and some videos had something odd going on with their frame count. These problems are listed in the exceptions.txt file.

Future Plans on VideoBadminton: I want to clean up some of the dataset, as there are files that I believe to be misclassified, and I want to figure out the exceptions. Furthermore, I plan on training various shot classification models using this data.


<img width="584" alt="Screenshot 2025-06-12 at 2 57 15 PM" src="https://github.com/user-attachments/assets/ba6224d1-72e0-4d8d-8294-a016e3f938cb" />
