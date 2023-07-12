from .models import (blister_user, lab_report, blisters,  predefined_precautions, prescription, tablets, tablets_user, lab_report_precautions)
from rest_framework import viewsets, permissions, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
# from rest_framework.views import APIView
from django.http import Http404
from .serializers import (blister_userSerializer, lab_reportSerializer, blistersSerializer, predefined_precautionsSerializer, 
                            prescriptionSerializer, tabletsSerializer, tablets_userSerializer, lab_report_precautionsSerializer)
from rest_framework import status
from rest_framework import mixins

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

import cv2
import numpy as np
import urllib.request
import pytesseract
import json
from PIL import Image

from models.MLmodels import blisterML
from models.MLmodels import pill_dentifier_update as pill_dentifier

from models.MLmodels.prescription import predict_result_update as prescription_predict_result
from models.MLmodels import lab_report as lab_reportttt
import os

class tablets_userList(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
    """
    List all tablets_users, or create a new tablets_user.
    """
    serializer_class = tablets_userSerializer
    queryset = tablets_user.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):

        created_done = self.create(request, *args, **kwargs)
        req = urllib.request.urlopen(created_done.data['image_url'])
        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(img,-1)

        img = cv2.resize(img, (224, 224))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.array(img/255)
        img = np.expand_dims(img, 0)
        # # img = img/255
        img = pill_dentifier.predict(img)
        print(img)
        created_done.data['predictedout'] = img
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return created_done

class tablets_userDetail(mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin,
                            generics.GenericAPIView):
    """
    Retrieve, update or delete a tablets_user instance.
    """
    serializer_class = tablets_userSerializer
    queryset = tablets_user.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

class tabletsList(APIView):
    """
    List all tabletss, or create a new tablets.
    """
    def get(self, request, format=None):
        tabletss = tablets.objects.all()
        serializer = tabletsSerializer(tabletss, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = tabletsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class tabletsDetail(APIView):
    """
    Retrieve, update or delete a tablets instance.
    """
    def get_object(self, pk):
        try:
            return tablets.objects.get(pk=pk)
        except tablets.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        tablets = self.get_object(pk)
        serializer = tabletsSerializer(tablets)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        tablets = self.get_object(pk)
        serializer = tabletsSerializer(tablets, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        tablets = self.get_object(pk)
        tablets.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class prescriptionList(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
    """
    List all prescriptions, or create a new prescription.
    """
    serializer_class = prescriptionSerializer
    queryset = prescription.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):

        created_done = self.create(request, *args, **kwargs)
        req = urllib.request.urlopen(created_done.data['image_url'])
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        # img = img.resize((32, 128))
        # img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (32, 128))
        # img = np.array(img)
        # img = np.expand_dims(img, 0)
        # img = img/255

        values = prescription_predict_result(img)
        # image read done here
        # now add the ML model here to pass the image and get the result
        print(values)
        created_done.data['predictedout'] = values
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return created_done

class prescriptionDetail(mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin,
                            generics.GenericAPIView):
    """
    Retrieve, update or delete a prescription instance.
    """
    serializer_class = prescriptionSerializer
    queryset = prescription.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

class predefined_precautionsList(APIView):
    """
    List all predefined_precautionss, or create a new predefined_precautions.
    """
    def get(self, request, format=None):
        predefined_precautionss = predefined_precautions.objects.all()
        serializer = predefined_precautionsSerializer(predefined_precautionss, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = predefined_precautionsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class predefined_precautionsDetail(APIView):
    """
    Retrieve, update or delete a predefined_precautions instance.
    """
    def get_object(self, pk):
        try:
            return predefined_precautions.objects.get(pk=pk)
        except predefined_precautions.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        predefined_precautions = self.get_object(pk)
        serializer = predefined_precautionsSerializer(predefined_precautions)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        predefined_precautions = self.get_object(pk)
        serializer = predefined_precautionsSerializer(predefined_precautions, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        predefined_precautions = self.get_object(pk)
        predefined_precautions.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class blistersList(APIView):
    """
    List all blisterss, or create a new blisters.
    """
    def get(self, request, format=None):
        blisterss = blisters.objects.all()
        serializer = blistersSerializer(blisterss, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = blistersSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class blistersDetail(APIView):
    """
    Retrieve, update or delete a blisters instance.
    """
    def get_object(self, pk):
        try:
            return blisters.objects.get(pk=pk)
        except blisters.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        blisters = self.get_object(pk)
        serializer = blistersSerializer(blisters)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        blisters = self.get_object(pk)
        serializer = blistersSerializer(blisters, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        blisters = self.get_object(pk)
        blisters.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class lab_reportList(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
    """
    List all lab_reports, or create a new lab_report.
    """
    serializer_class = lab_reportSerializer
    queryset = lab_report.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):

        created_done = self.create(request, *args, **kwargs)

        req = urllib.request.urlopen(created_done.data['image_url'])
        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
        
        img = cv2.imdecode(img,-1)
        img = cv2.resize(img, (800, 800))
        img = np.array(img)
        
        im = Image.fromarray(img)

        pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/New folder/tesseract.exe'
        xml = pytesseract.image_to_string(im)
        print(xml)
        all_data, val1, val2, val3, val4, val5 = lab_reportttt.lab_report_precautions(xml)
        # print(data)
        created_done.data['item1'] = val1
        created_done.data['item2'] = val2
        created_done.data['item3'] = val3
        created_done.data['item4'] = val4
        created_done.data['item5'] = val5

        # lab report get a specific lab report.
        lab_report.objects.filter(
            id=created_done.data['id']).update(
                item1=val1, item2=val2, item3=val3, item4=val4, item5=val5
                )
        return created_done


class lab_reportDetail(mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin,
                            generics.GenericAPIView):
    """
    Retrieve, update or delete a lab_report instance.
    """
    serializer_class = lab_reportSerializer
    queryset = lab_report.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
    

class blister_userList(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
    """
    List all blister_users, or create a new blister_user.
    """
    serializer_class = blister_userSerializer
    queryset = blister_user.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):

        created_done = self.create(request, *args, **kwargs)
        req = urllib.request.urlopen(created_done.data['image_url'])
        img = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(img,-1)

        img = cv2.resize(img, (224, 224))
        img = np.array(img)
        img = np.expand_dims(img, 0)
        # img = img/255
        img = blisterML.predict_result(img)
        # print(img)
        created_done.data['predictedout'] = img
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return created_done

class blister_userDetail(mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin,
                            generics.GenericAPIView):
    """
    Retrieve, update or delete a blister_user instance.
    """
    serializer_class = blister_userSerializer
    queryset = blister_user.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)



class lab_report_precautionsList(mixins.ListModelMixin, mixins.CreateModelMixin, generics.GenericAPIView):
    """
    List all lab_report_precautions, or create a new blister_user.
    """
    serializer_class = lab_report_precautionsSerializer
    queryset = lab_report_precautions.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):

        created_done = self.create(request, *args, **kwargs)
        return created_done

class lab_report_precautionsDetail(mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            mixins.DestroyModelMixin,
                            generics.GenericAPIView):
    """
    Retrieve, update or delete a blister_user instance.
    """
    serializer_class = lab_report_precautionsSerializer
    queryset = lab_report_precautions.objects.all()
    permission_classes = [
        permissions.IsAuthenticated,
    ]

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

@api_view(('GET',))
def lab_report_details(request,pk):

    values = lab_report.objects.filter(user=pk).order_by('-id')[:6]
    cholesterol = []
    HDL = []
    LDL = []
    TG = []
    glucose = []
    for i in values:
        cholesterol.append(float(i.item1))
        HDL.append(float(i.item2))
        LDL.append(float(i.item3))
        TG.append(float(i.item4))
        glucose.append(float(i.item5))


    return Response({'cholesterol':cholesterol,
                    'HDL':HDL,
                    'LDL':LDL,
                    'TG':TG,
                    'glucose':glucose})


@api_view(('GET',))
def get_tablet_details(request,pk):
    
    details = tablets.objects.filter(name=pk)
    print(details)
    serializer = tabletsSerializer(details, many=True)
    return Response(serializer.data)
