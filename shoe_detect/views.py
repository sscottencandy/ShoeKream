from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import UploadedImg, Product, ProductImg, Recommended
import pickle
import os
from .utils import image_crop, clip_shoes
from django.http import Http404
from django.urls import reverse
from django.db.models import Prefetch


file_path = os.getcwd()+'/models/model.pkl'
yolo_file_path = os.getcwd()+'/models/model_yolo.pkl'

#서버 2개 만들어서 하나는 yolo,하나는 clip돌리는 서버로 구축
#yolo 돌리는 서버에서는 input image를 받아서 label을 json형태로 return 하는 역할, API서버  

# Create your views here.
def index(request):
   

    input_image = UploadedImg.objects.all() 
    return render(request, "boards/index.html", {"input_image" : input_image})

def detail(request, id): #id = product db에서 만들어진 primary key
    ProdInfoBoards = Product.objects.filter().get(id = id)
    ProdImgBoard = ProductImg.objects.filter(prod = id).first()
    RecommendBoard = Recommended.objects.all()
    return render(request, "boards/detail.html", {
                                                    "ProdInfoBoards": ProdInfoBoards,
                                                    "ProdImgBoard": ProdImgBoard,
                                                    "RecommendBoard": RecommendBoard
                                                    })
import math
def shoe_list(request, id):
    # ProdInfoBoards = Product.objects.filter().get(id = id) #하나의 객체만 뽑기 때문에 not iterable하다. 
    ProdImgBoard = ProductImg.objects.filter(prod = id).first()
    RecommendBoard = Recommended.objects.all()
    UploadImgBoard = UploadedImg.objects.filter().get(id = id)
    
    # URL 쿼리 매개변수에서 cropped_result 추출
    query_params_str = request.GET.get('cropped_result')
    query_params = query_params_str.split("%") if query_params_str else []
    print(query_params)
    
    ProdInfoBoards = Product.objects.all().filter(name__in = query_params).prefetch_related('product_img_set')
    # print()
    # for prod_name in query_params:
    #     ProdInfoBoards.append()
    print(len(ProdInfoBoards))
    result = []
    data_total = math.ceil(len(ProdInfoBoards) / 4)
    
    for idx in range(data_total):
        if idx == data_total-1:
            result.append(ProdInfoBoards[idx*4:])
        else:
            result.append(ProdInfoBoards[idx*4 : (idx+1)*4])
    
    return render(request, "boards/shoe_list.html", { 
                                                    "ProdInfoBoards": result,
                                                    # "ProdImgBoard": ProdImgBoard,                                         
                                                    # "RecommendBoard": RecommendBoard,
                                                    # "UploadImgBoard":UploadImgBoard,
                                                    # "query_params": query_params
                                                     }
    )

from uuid import uuid4  # universal unique id
def crop_action(request):
    if request.method != 'POST':
        raise Http404
    image_file = request.FILES['chooseFile']
    image = image_file.read()
    
    # change image name to prevent duplicated filename
    ext =  image_file.name.split('.')[-1]
    image_file.name = str(uuid4()) + f'.{ext}'
    
    image_instance = UploadedImg(
        image = image_file
    )
    image_instance.save()
    
    filename = image_file.name
    cropped_result = clip_shoes(image_crop(image, filename, 0.75))
    
    # print(cropped_result)
    tmp = '%'.join(cropped_result)
    print('tmp: ', tmp)
    
    return redirect(reverse('shoe_list', kwargs={"id": str(image_instance.id)}) + f"?cropped_result={tmp}")

    # return redirect(f"/shoe_detect/shoe_list/{image_instance.id}" ) #업로드한 이미지의 id가 들어와야됨(db에 저장된 uploadedImg)

