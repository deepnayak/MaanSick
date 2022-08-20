from django.shortcuts import render
from django.http import HttpResponse  
from portal.functions.functions import handle_uploaded_file    
from portal.form import StudentForm  
# from django.http import HttpResponse
# Create your views here.

def index(request):
    if request.method == 'POST':  
        student = StudentForm(request.POST, request.FILES)  
        if student.is_valid():  
            handle_uploaded_file(request.FILES['file'])  
            return HttpResponse("File uploaded successfuly")  
    else:  
        student = StudentForm()  
        return render(request,'portal/index.html',{'form':student})   
    # return render(request, 'portal/home.html')

