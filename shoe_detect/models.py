from django.db import models

# Create your models here.
class UploadedImg(models.Model): #사용자가 업로드한 이미지
  id = models.AutoField(primary_key=True) #고유 key값
  uploaded_at = models.DateTimeField(auto_now=True) #업로드한 날짜(시간)
  # img_path = models.TextField() #이미지 경로
  image = models.ImageField(upload_to='./') #사진 파일 자체
  

"""
img = UploadedImg.objects.get(id=id)
img.image.path
""" 

class Cropped(models.Model): #사용자가 업로드한 이미지에서 신발만 검출한 이미지
  id = models.AutoField(primary_key=True) 
  uploaded_img_id = models.ForeignKey(UploadedImg, on_delete=models.CASCADE) #사용자가 업로드한 이미지의 key를 참조 
  cropped_img = models.TextField() #잘라낸 이미지의 id?
  
class Product(models.Model): #추천해 줄 상품 정보 list
  id = models.AutoField(primary_key=True)
  name = models.CharField(max_length=256)
  price = models.IntegerField(null = True)
  # original_price = models.IntegerField(null = True)
  date_release = models.DateField( null = True)
  brand = models.CharField(max_length=50, null = True)
  prod_id = models.CharField(max_length=128, null = True)
  original_price_currency	= models.CharField(max_length=64, null = True)
  original_price_with_currency = models.CharField(max_length=64, null = True)
  krw = models.CharField(max_length=64, null = True)
  
  @property
  def main_img(self):
    return self.product_img_set.first() #type:ignore
  
class Recommended(models.Model): #사용자가 찾는 신발과 유사한 제품 
  prod = models.ForeignKey(Product, on_delete=models.CASCADE) #전체 상품 리스트에서 key를 참조 ->cropped_id와 같은 값이어야 함
  cropped_id = models.ForeignKey(Cropped, on_delete=models.CASCADE) #prod와 일치해야 함 
  click_count = models.IntegerField(null = True)

class ProductImg(models.Model): #이미지만 띄워주는 DB
  id = models.AutoField(primary_key=True)
  prod = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='product_img_set')
  img_path = models.CharField(max_length=255)
  # models.ImageField()
    
