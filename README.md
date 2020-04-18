# HSI2020

This repository contains source code, data and paper submitted to [13th International Conference on Human System Interaction (HSI 2020)](http://hsi2020.welcometohsi.org) Tokyo, Japan, June 6-8, 2020.


<h1 align="center">
  Electric Wheelchair-Humanoid Robot Collaboration for Clothing Assistance of the Elderly
</h1>
</br>


<h1 align="center">
  <img src="data/logo/toyo.jpg" height="40px">
  <img src="data/logo/ieee_ies.jpg" height="40px">
  <img src="data/logo/ieee_ies_tchf.jpg" height="40px">
  <img src="data/logo/sig_aac.png" height="40px">
  </br>
  <sup>Conference Sponsors</sup>
</h1>


## Requirements for Paper Compilation
1. Make sure to have [TeX Live](https://www.tug.org/texlive/) installed.


## Info. on Paper Compilation
* **Linux Platform:** Please invoke shell script `sh compile.sh` from the terminal.
* **Windows Platform:** Please use any TeX editor.
* **Manual Compilation:** Make sure to follow the sequence of commands mentioned below to compile the file-
    ```
    1. pdflatex main.tex
    2. biber main
    3. pdflatex main.tex
    4. pdflatex main.tex
    ```


## ROS Package
The package, i.e., `baxter_whill_movement` is for Baxter and Whill cooperative movement to perform robotic clothing assistance.
The updated code should be available at [here](https://github.com/ravijo/baxter_whill_movement).


## Manifold Relevance Determination (MRD)
* **Training:** Please check [mrd_model.ipynb](https://github.com/ravijo/HSI2020/blob/master/scripts/mrd_model.ipynb) inside `scripts` subfolder.
* **Model:** Please check [unzip_me_please.zip](https://github.com/ravijo/HSI2020/tree/master/data/model) file. Don't forget to unzip it.


