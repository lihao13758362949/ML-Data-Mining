![在这里插入图片描述](https://img-blog.csdnimg.cn/20191011205351279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
## 2.1 sigmoid
$y=\frac{1}{1+e^{-z}}$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191020151239744.png)
## 2.2 tanh
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007202303262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
$g(x)=tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$
$g'(x)=1-[g(x)]^2$
## 2.3 relu
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007215802691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
$y=max(0,z)$

## 2.4 softmax
$\operatorname{softmax}(z)_{i}=\frac{\exp \left(z_{i}\right)}{\sum_{j} \exp \left(z_{j}\right)}$
对多类别分类，概率和为1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017183024308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjI5Nzg1NQ==,size_16,color_FFFFFF,t_70)
