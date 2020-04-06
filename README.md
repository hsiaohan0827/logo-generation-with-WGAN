# logo-generation-with-WGAN

train GAN: 
```
python3 main.py
```

In line 101(```if __name__=="__main__":```)中更改參數

要在這個資料夾下再放一個資料夾再放圖 ex: 圖放在./LLD/LLD1/, dataroot = './LLD'
```
dataroot = path to database
path_res = path to output ckp and config
```

generate images: 
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
