# Programming Generative AI

[![Twitter Follow badge](https://img.shields.io/twitter/follow/jonathandinu)](https://twitter.com/jonathandinu)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCi0Hd3U6xb4V0ApUhAIfu9Q?color=%23FF0000&logo=youtube&style=flat-square)](https://www.youtube.com/channel/UCi0Hd3U6xb4V0ApUhAIfu9Q)

> 18+ hours of video taking you all the way from VAEs to near real-time Stable Diffusion with PyTorch and Hugging Face... with plenty of hands-on examples to make deep learning fun again!

This repository contains the code, slides, and examples from my [Programming Generative AI](https://www.oreilly.com/videos/programming-generative-ai/9780135381090/) video course.

## Overview

_Programming Generative AI_ is a hands-on tour of deep generative modeling, taking you from building simple feedforward neural networks in PyTorch all the way to working with large multimodal models capable of simultaneously understanding text and images. Along the way, you will learn how to train your own generative models from scratch to create an infinity of images, generate text with large language models similar to the ones that power applications like ChatGPT, write your own text-to-image pipeline to understand how prompt-based generative models actually work, and personalize large pretrained models like Stable Diffusion to generate images of novel subjects in unique visual styles (among other things).

## Course Materials

The code, slides, and exercises in this repository are (and will always be) freely available. The corresponding videos can be purchased on:

- [InformIT](https://www.informit.com/store/programming-generative-ai-video-course-9780135381151): individual à la carte purchase (40% off with code: __VIDEO40__)
- [O'Reilly Learning](https://www.oreilly.com/videos/programming-generative-ai/9780135381090/): monthly subscription

The easiest way to get started (videos or not) is to use a cloud notebook environment/platform like [Google Colab](https://colab.google/) (or Kaggle, Paperspace, etc.). For convenience I've provided links to the raw Jupyter notebooks for local development, a [NBViewer](https://nbviewer.org/) link if you would like to browse the code without cloning the repo (or you can use the built-in Github viewer), and a Colab link if you would like to interactively run the code without setting up a local development environment (and fighting with CUDA libraries).

| Notebook                                                                                          |                                                                                                     NBViewer (static)                                                                                                      |                                                                                          Google Colab (interactive)                                                                                           |
| :------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Lesson 1: The What, Why, and How of Generative AI](01-intro-google-colab.ipynb)                  |      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/01-intro-google-colab.ipynb)      |                                   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/01-intro-google-colab.ipynb)                                    |
| [Lesson 2: PyTorch for the Impatient](02-pytorch-for-the-impatient.ipynb)                         |  [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/02-pytorch-for-the-impatient.ipynb)   |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/02-pytorch-for-the-impatient.ipynb)   |
| [Lesson 3: Latent Space Rules Everything Around Me](03-latent-space-vae.ipynb)                    |       [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/03-latent-space-vae.ipynb)       |       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/03-latent-space-vae.ipynb)       |
| [Lesson 4: Demystifying Diffusion](04-demystifying-diffusion.ipynb)                               |    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/04-demystifying-diffusion.ipynb)    |    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/04-demystifying-diffusion.ipynb)    |
| [Lesson 5: Generating and Encoding Text with Transformers](05-generating-text-transformers.ipynb) | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/05-generating-text-transformers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/05-generating-text-transformers.ipynb) |
| [Lesson 6: Connecting Text and Images](06-connecting-text-images.ipynb)                           |    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/06-connecting-text-images.ipynb)    |    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/06-connecting-text-images.ipynb)    |
| [Lesson 7: Post-Training Procedures for Diffusion Models](07-post-training-diffusion.ipynb)       |   [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/jonathandinu/programming-generative-ai/blob/main/07-post-training-diffusion.ipynb)    |   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/programming-generative-ai/blob/main/07-post-training-diffusion.ipynb)    |


> If you find any errors in the code or materials, please open a [Github issue](https://github.com/jonathandinu/programming-generative-ai/issues) or email [errata@jonathandinu.com](mailto:errata@jonathandinu.com).

## Selected Examples

These are some of my favorite examples from the course of what cool things you can do with Generative AI, to give you a taste of what you can expect.

Coming Soon....

###

###


## Skill Level

Intermediate to advanced

## Learn How To

- Train a variational autoencoder (VAE) with PyTorch to learn a compressed latent space of images.
- Generate and edit realistic human faces with unconditional diffusion models and SDEdit.
- Use large language models such as GPT2 to generate text with Hugging Face Transformers.
- Perform text-based semantic image search using multimodal models such as CLIP.
- Program your own text-to-image pipeline to understand how prompt-based generative models such as Stable Diffusion actually work.
- Properly evaluate generative models, both qualitatively and quantitatively.
- Automatically caption images using pretrained foundation models.
- Generate images in a specific visual style by efficiently fine-tuning Stable Diffusion with LoRA.
- Create personalized AI avatars by teaching pretrained diffusion models new subjects and concepts with Dreambooth.
- Guide the structure and composition of generated images using depth and edge conditioned ControlNets.
- Perform near real-time inference with SDXL Turbo for frame-based video-to-video translation.

## Who Should Take This Course

- **Engineers** and **developers** interested in building generative AI systems and applications.
- **Data scientists** interested in working with state-of-the-art deep learning models.
- **Students, researchers, and academics** looking for an applied or hands-on resource to complement their theoretical or conceptual knowledge.
- **Technical artists** and **creative coders** who want to augment their creative practice.
- **Anyone** interested in working with generative AI who does not know where or how to start.

## Prerequisites

- Comfortable programming in Python
- Knowledge of machine learning basics
- Familiarity with deep learning and neural networks will be helpful but is not required

## Lesson Descriptions

### Lesson 1: The What, Why, and How of Generative AI

Lesson 1 starts off with an introduction to what generative AI actually is, at least as it's relevant to this course, before moving into the specifics of deep generative modeling. It covers the plethora of possible multimodal models (in terms of input and output modalities) and how it is possible for algorithms to actually generate rich media seemingly out of thin air. The lesson wraps up with a bit of the formalization and theory of deep generative models, and the tradeoffs between the various types of generative modeling architectures.

### Lesson 2: PyTorch for the Impatient

Lesson 2 begins with an introduction to PyTorch and deep learning frameworks in general. Jonathan shows you how the combination of automatic differentiation and transparent computation on GPUs have really enabled the current explosion of deep learning research. He also shows you how you can use PyTorch to implement and learn a linear regression model as a stepping stone to building much more complex neural networks. Finally, the lesson demonstrates how to combine all of the components that PyTorch provides to build a simple feedforward multi-layer perceptron.

### Lesson 3: Latent Space Rules Everything Around Me

Lesson 3 starts with a primer on how computer programs actually represent images as tensors of numbers. Jonathan covers the details of convolutional neural networks and the specific architectural features that enable computers “to see”. Next, you get your first taste of latent variable models by building and training a simple autoencoder to learn a compressed representation of input images. At the end of the lesson, you encounter your first proper generative model by adding probabilistic sampling to the autoencoder architecture to arrive at the variational autoencoder (VAE)—a key component in future generative models that we will encounter.

### Lesson 4: Demystifying Diffusion

Lesson 4 begins with a conceptual introduction to diffusion models, a key component in current state of the art text-to-image systems such as Stable Diffusion. Lesson 4 is your first real introduction to the Hugging Face ecosystem of open-source libraries, where you will see how we can use the Diffusers library to generate images from random noise. The lesson then slowly peels back the layers on the library to deconstruct the diffusion process and show you the specifics of how a diffusion pipeline actually works. Finally, you learn how to leverage the unique affordances of a diffusion model’s iterative denoising process to interpolate between images, perform image-to-image translation, and even restore and enhance images.

### Lesson 5: Generating and Encoding Text with Transformers

Just as Lesson 4 was all about images, Lesson 5 is all about text. It starts with a conceptual introduction to the natural language processing pipeline, as well as an introduction to probabilistic models of language. You then learn how you can convert text into a representation more readily understood by generative models and explore the broader utility of representing words as vectors. The lesson ends with a treatment of the transformer architecture, where you will see how you can use the Hugging Face Transformers library to perform inference with pre-trained large language models (LLMs) to generate text from scratch.

### Lesson 6: Connecting Text and Images

Lesson 6 starts off with a conceptual introduction to multimodal models and the requisite components needed. You see how contrastive language image pre-training jointly learns a shared model of images and text, and learn how that shared latent space can be used to build a semantic image search engine. The lesson ends with a conceptual overview of latent diffusion models, before deconstructing a Stable Diffusion pipeline to see precisely how text-to-image systems can turn a user supplied prompt into a never-before-seen image.

### Lesson 7: Post-Training Procedures for Diffusion Models

Lesson 7 is all about adapting and augmenting existing pre-trained multimodal models. It starts with the more mundane, but exceptionally important, task of evaluating generative models before moving on to methods and techniques for parameter efficient fine tuning. You then learn how to teach a pre-trained text-to-image model such as Stable Diffusion about new styles, subjects, and conditionings. The lesson finishes with techniques to make diffusion much more efficient to approach near real-time image generation.

## Copyright Notice and License

©️ 2024 Jonathan Dinu. All Rights Reserved. Removal of this copyright notice or reproduction in part or whole of the text, images, and/or code is expressly prohibited.

For permission to use the content in your own presentation (blog posts, lectures, videos, courses, etc.) please contact copyright@jonathandinu.com.
