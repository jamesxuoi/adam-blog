---
layout: post
title:  TinyML Speech Recognition for Virtual Assistant, Part 1
date:   2020-02-15 13:32:20 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: post-1.jpg # Add image post (optional)
tags: [Blog, NLP, TinyML, Machine Learning]
author: James Xuoi # Add name author (optional)
--- 

After reading one of my 2019 favourite books, ['TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers'](https://learning.oreilly.com/library/view/tinyml/9781492052036/) by *Daniel Situnayake* and *Pete Warden*. I decided to build a TinyML project for my final Capstone at Institute of Data, where I later got hired to the Data Science Training Team instantly after presenting the Speech Recognition project. In this post, I'll introduce the concept of TinyML, why it is a giant opportunity as well as step-by-step guidance to build your personal TinyML Speech Recognition Virtual Assistant.

![TinyML]({{site.baseurl}}/assets/img/post_1/tinyml.jpg)


### 1. What is TinyML?

The ‘Tiny’ here mentions about the Ultra-Low-Power microcontrollers and ‘ML’ simply stands for Machine Learning. As a combination, TinyML is an emerging concept to run Machine Learning inference with TensorFlow Lite on the Ultra Low-Power microcontrollers such as: Arduino, Sparkfun Edge and STMicroelectronics boards. 

This ideal is developed by the OK Google team initially in 2014. They were running a 14 kilobytes (KB) deep neural networks on the digital signal processors (DSPs) present in most Android phones, which continuously listening for the “OK Google” wake words. In addition, these DSPs had only tens of kilobytes of RAM and flash memory; and these specialized chips use only a few milliwatts (mW) of power. The purpose of TinyML is for software and hardware developers who want to build embedded systems using machine learning and never worry about memory, battery or operating speed.

### 2. Why TinyML is the future?

> Tiny Computers are Already Cheap and Energy Efficiency

Microcontrollers (or MCUs) are packages containing a small CPU with possibly just a few kilobytes of RAM, and are embedded in consumer, medical, automotive and industrial devices. They are designed to use very small amounts of energy, and to be cheap enough to include in almost any object that’s sold, with average prices expected to dip below 50 cents this year. The biggest benefit for the manufacturer is that standard controllers can be programmed with software rather than requiring custom electronics for each task, so they make the manufacturing process cheaper and easier. A microcontroller itself might only use a milliwatt or even less, a coin battery might have 2,500 Joules of energy to offer, so even something drawing at one milliwatt will only last about a month. Of course most current products use duty cycling and sleeping to avoid being constantly on, but you can see what a tight budget there is even then.

> Deep Learning Runs Well on Existing MCUs

Microcontrollers (MCUs) tend to be less expensive than, simpler to set-up, and simpler to operate than microprocessors (MPUs). The comparatively low memory requirements (just tens or hundreds of kilobytes) also mean that lower-power SRAM or flash can be used for storage. This makes deep learning applications well-suited for microcontrollers, especially when eight-bit calculations are used instead of float, since MCUs often already have DSP-like instructions that are a good fit. This idea isn’t particularly new, both Apple and Google run always-on networks for voice recognition on these kind of chips, but not many people in either the ML or embedded world seem to realize how well deep learning and MCUs match.

> TinyML Makes Sense of Sensor Data

In the last few years its suddenly become possible to take noisy signals like images, audio, or accelerometers and extract meaning from them, by using neural networks. Because we can run these networks on microcontrollers, and sensors themselves use little power, it becomes possible to interpret much more of the sensor data we’re currently ignoring. For example, every morning an engineer needs to walk along the row of machines to check which will have to be taken offline for servicing. If every machine could be attached a battery-powered accelerometer and microphone that would learn usual operation and signal if there was an anomaly, you might be able to catch issues before they became real problems. There are probably a hundred products I could dream up, but running neural networks on voice recognition is the main project I’d like to explain details in this post. Let’s get it started!

### 3. Building TinyML Speech Recognition.

In the TinyML book, Peter Warden covers about the Tensorflow Lite conversion and deployment of Speech Recognition model on different microcontroller chips (source code can be found on Tensorflow Github page). However, before deployment, the detail instructions of data extraction, data analytics and model building aren't listed there. Hence, the steps below will fully cover from extracting .wav files data to deploying neural networks on an Arduino board. (code can be found on my Github page)

> Materials:

*- A laptop with Windows/ Mac/ Linux/ Ubuntu operating system.*

*- An Arduino BLE board*

*- Micro USB cable*

The audio files are collected from a research project of Google TensorFlow named ‘Simple Audio Recognition’. In detail, the archive Speech Commands data set is over 2GB and contains 36 different key words in 108,000 WAVE audio files in the list below:

'four', 'forward', 'off', 'five', 'on', 'six', 'down', "house', 'two', 'visual', 'up', 'zero', 'three', 'stop','follow', "happy', 'backward', 'learn', 'cat', 'right', 'eight', 'sheila', 'nine', 'yes', 'one', 'no', 'left','tree', 'bed', 'bird, 'go', 'wow', 'seven', 'marvin', 'dog', "background_noise'.

This data was collected by Google and released under a CC BY license, and you can help improve it by contributing five minutes of your own voice at the on-going Open Speech Recording project. The audio file is mainly selected to be clear and clean for training purposes, therefore audio noises has been reduced or eliminated.

![speech_recognition_overview]({{site.baseurl}}/assets/img/post_1/speech_recognition_overview.jpg)
<center>Figure 1: An overview of data distribution</center>

> Download and Extract

Download the speech-commands data set from:
https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.0.2.tar.gz


Extract tar.gz file
``` python
import tarfile
tar = tarfile.open('.../data_speech_commands_v0.02.tar.gz', "r:gz")
tar.extractall(path='extracting_folder_path')
tar.close()
```


> Data Analytics and Visualisation

After setting up a data pipeline to load all '.wav' files, an audio file is randomly chosen for data analysing and visualising purposes.

In this case, 'two' is chosen. Let's quickly listen to it!

<audio src="{{site.baseurl}}/assets/media/617aeb6c_nohash_3.wav" controls preload></audio>

What you have just heard is a sound of 16 000 Hz frequencies. According to the official frequency chart, the “perfect” human ear can hear frequencies ranging from 20Hz to 20 000 Hz. In addition, nowadays most music's frequencies can be found around 50Hz and 16 000 Hz; and the visualisations below show the Digital Signal Processing (DSP) of 'Two' in various forms.

**Sound Wave Form:** This is the most common seen digital signal form, whenever a song is played then the screen of your device will automatically show the audio sound wave's amplitude varies through time.

![Sound_Wave_Form]({{site.baseurl}}/assets/img/post_1/1.jpg)
<center>Figure 2: Sound Wave Form's amplitude varies through time</center>

*to be continued...*
