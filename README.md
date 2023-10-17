# MomoTranslator
Pure OpenCV comic translation tool

![Commit activity](https://img.shields.io/github/commit-activity/m/alicewish/MomoTranslator)
![License](https://img.shields.io/github/license/alicewish/MomoTranslator)
![Contributors](https://img.shields.io/github/contributors/alicewish/MomoTranslator)

简体中文 | [English](README_EN.md)

## 功能

漫画翻译辅助软件MomoTranslator当前的功能为：

1. 查找漫画画格

2. 查找漫画气泡

3. 根据画格排序气泡，也可手动调整气泡排序

4. 识别文字

5. 使用谷歌和ChatGPT网页版翻译

填字功能暂不公开，如果是汉化者要用可以找我跑填字脚本。

## 演示视频

之前的功能演示视频见https://www.bilibili.com/video/BV1P54y1Q7fW/

软件特点是主要功能基于opencv，可以不需要pytorch，也可以使用pytorch来提高精确度。

之前不公开代码的原因是遭受多名视频讲漫up的抄袭、性骚扰、网暴，我担心自己写的代码开源后成为逼死自己的工具。

由于现在ChatGPT的发展，我觉得也许有其他方法避免最糟糕的结果，可以先试试开源。

## 安装

首先，确保您的计算机上已安装Python 3.9或更高版本。您可以通过在命令行输入以下命令来检查Python版本：
```bash
python --version
```

克隆此存储库或下载ZIP文件并解压缩。

使用以下命令安装必需的Python库：
```bash
pip install -r requirements.txt
```

## 使用

在命令行中，进入到项目文件夹并运行pyqt5_momotranslator.py文件：

```bash
python pyqt5_momotranslator.py
```

运行程序后，将出现一个图形用户界面。

## 许可证

本项目根据MIT许可证授权。有关更多信息，请查看[LICENSE](https://github.com/alicewish/MomoTranslator/blob/main/LICENSE)文件。
