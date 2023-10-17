# MomoTranslator
Pure OpenCV comic translation tool

![Commit activity](https://img.shields.io/github/commit-activity/m/alicewish/MomoTranslator)
![License](https://img.shields.io/github/license/alicewish/MomoTranslator)
![Contributors](https://img.shields.io/github/contributors/alicewish/MomoTranslator)

## Features

MomoTranslator, a comic translation assistant tool, currently has the following capabilities:

1. Locating comic panels.
    
2. Identifying speech bubbles within the panels.
    
3. Ordering speech bubbles based on their panels; this order can be manually adjusted as needed.
    
4. Recognizing text within the speech bubbles.
    
5. Translating text using Google and ChatGPT web.
    
The text-filling feature is not publicly available at the moment. However, translators interested in this feature can contact me directly to run the text-filling script.

## Demo Video

For a demonstration of the tool's previous features, please visit [this video](https://www.bilibili.com/video/BV1P54y1Q7fW/).

One of the key characteristics of the software is its reliance on OpenCV for most functionalities. While PyTorch is not necessary, integrating it could enhance accuracy.

The reason behind not disclosing the code previously was due to incidents of plagiarism by several comic narrators on video platforms, sexual harassment, and cyberbullying. There was a concern that making the source code public could exacerbate these issues.

However, with the recent advancements in ChatGPT, there might be new methods to circumvent the worst-case scenarios, warranting an exploration into open-sourcing the tool.

## Installation

First, ensure that you have Python 3.9 or a later version installed on your computer. You can check your Python version by entering the following command in the command line:

`python --version` 

Clone this repository or download the ZIP file and extract it.

Install the necessary Python libraries with the following command:

`pip install -r requirements.txt` 

## Usage

In the command line, navigate to the project folder and execute the file pyqt5\_momotranslator.py:

`python pyqt5_momotranslator.py` 

Upon running the program, a graphical user interface will appear.

## License

This project is licensed under the terms of the MIT license. For more information, please refer to the [LICENSE](https://github.com/alicewish/MomoTranslator/blob/main/LICENSE) file.