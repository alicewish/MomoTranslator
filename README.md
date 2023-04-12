# MomoTranslator
Pure OpenCV comic translation tool

![Commit activity](https://img.shields.io/github/commit-activity/m/alicewish/MomoTranslator)
![License](https://img.shields.io/github/license/alicewish/MomoTranslator)
![Contributors](https://img.shields.io/github/contributors/alicewish/MomoTranslator)


用GPT4重构了我的漫画翻译辅助软件MomoTranslator


重构版当前的功能为：

1、查找漫画画格

2、查找漫画气泡

3、根据画格排序气泡后识别文字

4、翻译


嵌字功能尚未重构完成，故不放入。

另外，我也需要观察一下公开代码后是否会出现滥用，比如生成质量糟糕的“汉化”漫画和视频等。


之前的功能演示视频见https://www.bilibili.com/video/BV1P54y1Q7fW/

重构完嵌字功能我会做个新演示视频。


软件特点是主要功能基于opencv，可以不需要pytorch。

也可以使用pytorch来提高精确度。


之前不公开代码的原因是遭受多名视频讲漫up的抄袭、性骚扰、网暴，我担心自己写的代码开源后成为逼死自己的工具。

由于现在ChatGPT的发展，我觉得也许有其他方法避免最糟糕的结果，可以先试试开源。


原版软件（默墨汉化）有完整的在Photoshop中涂白、嵌字并导出为PSD的功能，此功能正在重构中，此外的需求可在issue里提。

# 安装

首先，确保您的计算机上已安装Python 3.9或更高版本。您可以通过在命令行输入以下命令来检查Python版本：
```bash
python --version
```

克隆此存储库或下载ZIP文件并解压缩。

使用以下命令安装必需的Python库：
```bash
pip install -r requirements.txt
```

# 使用

在命令行中，进入到项目文件夹并运行pyqt5_momotranslator.py文件：

```bash
python pyqt5_momotranslator.py
```

运行程序后，将出现一个图形用户界面。



# 许可证

MIT License
