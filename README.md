# logo-generation-with-WGAN

# train GAN 
```
python3 main.py
```

In line 101(```if __name__=="__main__":```)中更改參數

```
dataroot = path to database
path_res = path to output ckp and config
```
download logo data from LLD (Large Logo Dataset: https://data.vision.ee.ethz.ch/sagea/lld/)

data要放在這個資料夾下的子資料夾 ex: 圖放在./LLD/LLD1/, dataroot = './LLD'

# generate images
```
python3 generate.py
```

In line 20(```if __name__=="__main__":```)中更改參數
```
config = path to config
weights = path to ckp
output_dir = path to output images
nimages = number of images
```
